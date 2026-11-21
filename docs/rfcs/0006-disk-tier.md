# RFC 0006: A local SSD tier below the Host cache

Status: In progress. Builds on RFC 0004 (peer prefix sharing) and RFC 0005
(runtime P/D roles).

## Summary

Add the node's local NVMe as a third prefix-cache tier, `kDisk`, below the
Device and Host tiers. The tier lives in the same coordinator, the same
`BoundaryTree` and the same prefix indexes as the other two, so a page still
has exactly one owner and hybrid models (sliding windows, State checkpoints)
need no special case. The Host tier writes back to the Disk tier in the
background; a request whose prefix sits on Disk has it read into Host before
admission and then admits as an ordinary Host hit. Peer prefix sharing
extends to Disk: a holder pushes Disk pages straight into the reader's memory,
reading them with io_uring into a fixed staging ring and writing each chunk
over RDMA as soon as it is read.

The Disk tier never outlives its process: a restart reuses the file and
ignores its contents.

## Motivation

Agent workloads keep long sessions alive for minutes to hours. Host memory
holds a few hundred GB per node; the node's NVMe holds several TB and reads
at device speed (one data-center NVMe drive reads 7 GB/s at 256 KiB blocks
and 32 requests in flight, measured with this RFC's executor). Recomputing a
100K-token prefix of an 8B model takes seconds of prefill; reading its KV
from one drive takes about two.

Two properties make an external store a poor fit:

- **Dependencies.** A hybrid model resumes at a boundary only with the KV
  pages, the sliding window behind the boundary and every State checkpoint
  at it. Only the owner's `BoundaryTree` knows these dependencies; a flat
  key-value store evicts them independently (`cache-concepts.md`).
- **Write-back.** The Host tier already runs write-back; the coordinator's
  write-back and peer sharing are both incompatible with L3 storage
  (`SchedulerConfig::Validate`).

### Prior art

- **3FS USRBIO** submits I/O through a shared ring to the 3FS client process
  and reads into a registered shared buffer. It requires 3FS.
- **Mooncake's SSD offload** writes memory replicas to a local file and serves
  remote reads through the owner: the owner reads a batch into a buffer it
  leases to the requester, which then pulls it over RDMA and releases it.
  The SSD read and the network transfer are serial, and the lease adds two
  round trips. Pushing chunks to the requester's final buffer as they are
  read removes both; this RFC does that for peer reads.
- **SGLang HiCache** and **vLLM's offloading connector** treat SSD as one more
  key-value store under the Host tier.

## Design

### Terms

- **Clean / dirty**: a Host entry whose key is also published on Disk is
  clean; dropping it loses nothing. Any other Host entry is dirty.
- **Cleaning**: writing dirty Host entries to Disk ahead of need.
- **Prefetch**: reading a request's Disk-only prefix into Host before its
  admission.
- **Push**: a holder reading its Disk pages and writing them into a reader's
  memory.

### Invariants

1. **One owner, one tree.** Disk entries are registered in the owner's prefix
   indexes and hold `BoundaryTree` nodes like Device and Host entries. Gaps,
   dead entries and checkpoint dependencies are judged across all three
   tiers.
2. **The forward never waits for the SSD.** Disk feeds only Host. A request
   waits for its prefetch before admission, holding no Device pages and no
   request slot.
3. **Device write-back never waits for the SSD.** The Host tier keeps clean
   capacity by cleaning in the background; a Host placement drops clean
   entries before dirty ones.
4. **Nothing is published before it retired.** A Disk entry is published after
   every rank's write completed; a prefetched Host entry after every rank's
   read completed. Until then the operation holds its source and destination
   blocks.
5. **One path.** A prefetched prefix is an ordinary Host hit; a pushed peer
   page lands like any peer page.

### Tier model

```
 Device ──stream-ordered write-back──▶ Host ──cleaning (io_uring writes)──▶ Disk
   ▲                                     │  ▲                                │
   └──────layerwise H2D (Host hit)───────┘  └───prefetch (io_uring reads)────┘
```

The Disk tier requires Host write-back (`host_write_policy = kWriteBack`) and
excludes L3. Its `BlockPool` uses the Host packing, so a Disk block id maps to
a file offset exactly as a Host block id maps to a Host pool offset
(`parent * parent_bytes + child * block_bytes`): one block is one I/O and
nothing is repacked.

### Storage and I/O

Each cache-owning rank owns one preallocated file on local NVMe, opened with
`O_DIRECT`. There is no metadata file: the scheduler's index is the only
record, and a restart starts from an empty index. With the Disk tier on, the
Host layout rounds every group's block and the parent up to 4 KiB, the
direct-I/O alignment.

The engine host's Rust executor (`ts-disk`) moves aligned ranges between the
file and registered memory. It has two backends, chosen explicitly:

- **io_uring**: one thread owns a ring with the Host pool (and the push ring)
  registered as fixed buffers, in pieces of at most 1 GiB; no request crosses
  a piece. The file is not registered and SQPOLL is not used: the kernels and
  container policies in use refuse fixed files and unprivileged SQPOLL.
- **psync**: a pool of threads calling `pread`/`pwrite`, for hosts whose
  policy refuses io_uring.

Reads go ahead of writes. Writes pass a token bucket set to the configured
byte budget, which protects the drive's endurance, and take at most a bounded
share of the requests in flight. A job reports once, after every request it
issued retired; neither backend can stop a request the kernel already has, so
a job's blocks stay held until it reports.

### Cleaning

Every plan, if the Host tier's free plus clean capacity is below a watermark,
the scheduler picks the oldest dirty Host-only entries, in Host eviction
order, within the operation credits, and writes them to Disk blocks it
allocates (evicting Disk entries if needed). The write holds its Host sources
and Disk destinations; when every rank retired it, the Disk entries are
published with the Host entries' metadata (access epoch, logical position,
boundary kind) and the Host entries become clean. Entries already on Disk, a
peer's demoted chain and superseded checkpoint inputs are not written.

Cleaning copies; it never removes anything from Host, so a partly cleaned
segment opens no gap. A Host placement takes, in order: dead entries, entries
the Device holds, clean entries, then dirty entries by recency.

### Prefetch

A request with no Device or Host hit covering its cacheable prefix probes
Disk next: the keys the Disk index holds stand in as Host hits for the
Host-suffix probe (`ProbeHostSuffix`), the same hook L3 uses, so one matcher
picks the longest resumable boundary under every group's rule. The scheduler
allocates Host destinations for the keys Host lacks, holds the Disk sources,
and the request waits in `Prefetching` (holding no Device pages) until every
rank read them. The Host entries are then published, held for the request by
a Ready hold, and the request admits as an ordinary Host hit. A soft deadline
lets the request admit from its local hit while the read keeps its blocks. A
failed read removes the Disk entries it named.

### Hybrid attention

No hybrid-specific code: Disk follows the Host tier's rules.

- A State checkpoint is valid only at its exact boundary and only together
  with its history. With no history on any tier it is dead.
- A sliding group prefetches only the window behind the boundary.
- A tail is read and written whole. The tail table keeps a hint while any tier,
  or a write in flight, holds an entry for it.
- Replayable groups never reach Host, hence never Disk.
- `CheckpointTree` ranks checkpoints across the three tiers.

### Disk eviction

Free capacity first, then dead entries, then least recently used entries
(checkpoint retention rank included). Equal-age pages leave from the suffix,
so a parent page leaves after its children; a State checkpoint takes its
State siblings at the same boundary along. Entries a prefetch or a peer lease
holds are not evicted. Eviction does no I/O: it drops the index entry and the
slot is overwritten later.

### Peer sharing

**Holder.** A lookup probes Device, Host and Disk, and lists each page from
the fastest tier that holds it; `PeerPage::tier` may now be `kDisk`. A hard pin
holds Disk entries like the others. RFC 0004's argument that answers hold
until the next head applies unchanged: Disk entries are placed, evicted and
published only in plans.

Every rank runs a push endpoint, named in its layout beside its transfer
engine session; the layout also carries the Disk group layout (the Host
one). A PUSH names the lease, the holder's incarnation, the Disk block ids and
the reader's destinations. The holder rank reads each block with io_uring
into a fixed staging ring registered with the transfer engine
(FREE → SSD_READING → RDMA_WRITING → FREE) and writes it to the reader with the
transfer engine as soon as the read completes, so reads and network
transfers overlap. A block larger than a ring slot moves in slot-sized
chunks. A full ring applies backpressure. Neither the holder's scheduler nor
its forward thread takes part.

**Reader and landing tier.** `--peer-prefix-landing-tier` selects where a peer
prefix lands, with no default:

- `device`: RFC 0004's landing. Pages from the holder's Device or Host are
  read with one-sided reads; Disk pages are pushed into the same Device
  pages. The PUSH leaves at the point the reads start (after the plan's
  device work). The Device pages wait an SSD read longer.
- `host`: the whole peer span lands in Host blocks the plan allocates, as an
  L3 GET's destinations: Device and Host pages by one-sided reads, Disk pages
  by push. After the release vouches for them, they are published as Host
  entries under a Ready hold and the request admits as an ordinary Host hit.
  No Device pages wait for the transfer; the cost is Host capacity and one
  H2D copy.

Both share RFC 0004's lookup, pin and release, the whole-or-nothing landing
and the holder's vouching at release.

**Write fence.** A push makes the holder a remote writer into the reader's
memory, so a landing the reader gave up must not take late writes into pages
it reuses. The reader follows PD's fence (`event-loop.md`, PD receive
revocation): it sends CANCEL on the PUSH's ordered channel, and the holder
acknowledges once every write of the job retired or none was posted; the
holder posts no write past the job's deadline. Without an acknowledgment
within the budget the reader revokes the transfer engine's registration,
after which the NIC rejects late writes. Only RDMA revokes; elsewhere the
pages stay held.

**Capability.** A holder advertises its push endpoint; a reader says in its
lookup whether it accepts Disk pages. Without both, no Disk page is listed.

### Roles

Fused and P roles hold and read. The D role holds: it cleans the histories it
decoded to Disk for the next turn's P to read, and its remote admission keeps
the Device-plus-Host contract.

### Faults

- **A write fails**: nothing is published; the Host entry stays dirty.
- **A read fails**: the request admits from its local hit; the Disk entries
  read are removed.
- **A request never retires**: its blocks stay held for the process, and the
  executor stops admitting work once its queue cannot drain.
- **The holder's read fails during a push**: the landing gives up whole; the
  reader admits from its local hit.
- **A weight update or flush** clears the Disk index after every Disk
  operation retired; the file is not touched. A Disk operation is never
  retained across a clear, since a write finishing after it would publish old
  contents: one still in flight past the drain timeout fails the clear.

### Configuration

- `--disk-tier-paths`: one directory per drive; ranks' files are spread over
  them.
- `--disk-tier-size-gb`: Disk capacity per rank.
- `--disk-tier-io io_uring|psync`: the backend; required.
- `--disk-tier-write-mbps`: the write budget per rank; required.
- `--peer-prefix-landing-tier device|host`: required with peer sharing.

### Observability

Disk hit tokens, prefetch latency and failures, cleaning bytes and dropped
dirty entries, clean and dirty Host blocks, Disk occupancy and evictions,
executor queue depth and latency, write-budget waits; peer tokens by source
tier, push SSD time, ring waits, RDMA time and overlap, CANCEL and revocation
counts.

## Phases

1. `ts-disk` and a cold-read baseline.
2. The local Disk tier: three-tier index, cleaning, prefetch, the control
   plane's disk module, configuration.
3. Peer push: push endpoints and the staging ring, reader PUSH and CANCEL,
   the write fence, the Host landing tier, router credit for Disk sources.
4. Later: the D role's remote admission probing Disk, retraction images
   spilling to Disk, writing replicated payloads once per node.

## Testing

- Scheduler (C++): three-tier counts in `BoundaryTree`; cleaning on
  full-history, sliding and State group sets; clean-before-dirty Host
  placement; Disk eviction opens no gap; prefetch reads only what Host lacks;
  held Disk entries are not evicted; Disk pages exported and pinned for peers.
- Executor (Rust): byte round trips through both backends, alignment and
  bounds checks, write pacing, cancellation, close with queued and running
  work; push pipelining, backpressure, deadlines and CANCEL acknowledgments.
- GPU: outputs identical to recomputation with Disk hits, and with peer reads
  from Disk in both landing tiers; holder or reader killed mid-push.

## Open questions

1. The write budget, and whether only entries hit at least twice are worth
   writing.
2. Scheduler memory and planning cost with millions of Disk entries per rank.
3. Replicated payloads (MLA under attention TP) are written once per rank.
