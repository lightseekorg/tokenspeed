# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Standard-library reference consumer for TokenSpeed raw metrics JSONL files."""

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

FILE_NAME = re.compile(r"metrics-(.+)-(\d{6,})\.jsonl")


@dataclass(frozen=True)
class Cursor:
    filename: str
    offset: int
    sequence: int


class JsonlMetricsReader:
    """Consume retained samples, checkpointing only after the callback succeeds.

    Args:
        directory: One writer's directory on a local filesystem or shared volume.
        checkpoint: State file owned exclusively by this consumer.
        warn: Callback for missing files, sequence gaps and abandoned partial lines.
    """

    def __init__(
        self, directory: Path, checkpoint: Path, *, warn: Callable[[str], None]
    ):
        self.directory = directory.resolve()
        self.checkpoint = checkpoint.resolve()
        self.warn = warn
        self._cursors: dict[str, Cursor] = {}
        self._last_warnings: dict[tuple[str, str], str] = {}
        if not self.directory.is_dir():
            raise ValueError(f"Metrics directory does not exist: {self.directory}")
        if self.checkpoint.exists():
            data = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            if data["schema_version"] != 1 or data["directory"] != str(self.directory):
                raise ValueError(
                    "Checkpoint version or metrics directory does not match"
                )
            for stream, saved in data["streams"].items():
                cursor = Cursor(**saved)
                match = FILE_NAME.fullmatch(cursor.filename)
                if (
                    match is None
                    or match[1] != stream
                    or type(cursor.offset) is not int
                    or cursor.offset < 0
                    or type(cursor.sequence) is not int
                    or cursor.sequence < 0
                ):
                    raise ValueError(f"Invalid checkpoint cursor for stream {stream}")
                self._cursors[stream] = cursor

    def _warn(self, stream: str, kind: str, message: str) -> None:
        key = (stream, kind)
        if self._last_warnings.get(key) != message:
            self.warn(message)
            self._last_warnings[key] = message

    def _save(self, stream: str, cursor: Cursor) -> None:
        cursors = {**self._cursors, stream: cursor}
        data = {
            "schema_version": 1,
            "directory": str(self.directory),
            "streams": {key: asdict(value) for key, value in cursors.items()},
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.checkpoint.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                json.dump(data, output)
                output.write("\n")
            os.replace(temporary, self.checkpoint)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._cursors = cursors

    def drain(self, consume: Callable[[dict], None]) -> int:
        """Deliver complete records currently available and return their count.

        The callback must raise on delivery failure. Retrying after a crash or a
        checkpoint write failure can repeat records; use (stream_id, sequence)
        for deduplication. Streams are independent and have no global ordering.
        """
        streams: dict[str, list[tuple[int, Path]]] = {}
        for path in self.directory.iterdir():
            match = FILE_NAME.fullmatch(path.name)
            if match is not None:
                streams.setdefault(match[1], []).append((int(match[2]), path))

        delivered = 0
        for stream in sorted(streams.keys() | self._cursors.keys()):
            paths = sorted(streams.get(stream, []))
            cursor = self._cursors.get(stream)
            previous_shard = -1
            sequence = -1
            if cursor is not None:
                previous_shard = int(FILE_NAME.fullmatch(cursor.filename)[2])
                sequence = cursor.sequence
                if not any(path.name == cursor.filename for _, path in paths):
                    self._warn(
                        stream,
                        "missing",
                        f"Checkpoint file {cursor.filename} is missing; retained data may have a gap",
                    )
                paths = [
                    (shard, path) for shard, path in paths if shard >= previous_shard
                ]

            for index, (shard, path) in enumerate(paths):
                if shard > previous_shard + 1:
                    self._warn(
                        stream,
                        "shard",
                        f"Stream {stream}: missing shards before {path.name}",
                    )
                previous_shard = shard
                try:
                    source = path.open("rb")
                except FileNotFoundError:
                    self._warn(
                        stream, "removed", f"File {path.name} is missing during read"
                    )
                    continue
                with source:
                    # Bound each pass so a continuously growing stream cannot
                    # prevent discovery of other streams or a non-follow exit.
                    limit = os.fstat(source.fileno()).st_size
                    offset = (
                        cursor.offset if cursor and cursor.filename == path.name else 0
                    )
                    if offset > limit:
                        self._warn(
                            stream,
                            "truncated",
                            f"File {path.name} was truncated; rereading",
                        )
                        offset = 0
                    source.seek(offset)
                    while source.tell() < limit:
                        offset = source.tell()
                        line = source.readline(limit - offset)
                        if not line.endswith(b"\n"):
                            if index + 1 < len(paths):
                                self._warn(
                                    stream,
                                    "partial",
                                    f"Skipping incomplete tail of {path.name}; a later shard exists",
                                )
                            break
                        record = json.loads(line)
                        if (
                            not isinstance(record, dict)
                            or record.get("schema_version") != 1
                            or record.get("stream_id") != stream
                            or type(record.get("sequence")) is not int
                            or record["sequence"] < 0
                        ):
                            raise ValueError(
                                f"Invalid record in {path.name} at byte {offset}"
                            )
                        if record["sequence"] != sequence + 1:
                            self._warn(
                                stream,
                                "sequence",
                                f"Stream {stream}: sequence gap; expected {sequence + 1}, found {record['sequence']}",
                            )
                        consume(record)
                        cursor = Cursor(path.name, source.tell(), record["sequence"])
                        self._save(stream, cursor)
                        sequence = cursor.sequence
                        delivered += 1
        return delivered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--follow", action="store_true", help="Keep discovering new samples"
    )
    args = parser.parse_args()

    def emit(record: dict) -> None:
        sys.stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def warn(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr, flush=True)

    try:
        reader = JsonlMetricsReader(args.directory, args.checkpoint, warn=warn)
        while True:
            reader.drain(emit)
            if not args.follow:
                return 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
