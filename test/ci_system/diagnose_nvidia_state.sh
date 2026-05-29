#!/usr/bin/env bash
set -u

label="${1:-snapshot}"

section() {
  printf '\n'
  printf '=========================================================================\n'
  printf '  %s\n' "$1"
  printf '=========================================================================\n'
}

run_optional() {
  printf '$ %s\n' "$*"
  "$@" 2>&1 || true
}

section "NVIDIA diagnostics: ${label}"
run_optional date -u
run_optional hostname
run_optional uptime

section "nvidia-smi"
run_optional nvidia-smi

section "GPU query"
run_optional nvidia-smi \
  --query-gpu=index,name,uuid,pci.bus_id,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv

section "GPU compute apps"
run_optional nvidia-smi \
  --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv

section "GPU topology"
run_optional nvidia-smi topo -m

section "Memory"
run_optional free -h

section "CPU"
run_optional lscpu

section "Top processes by CPU"
ps -eo pid,ppid,user,stat,psr,pcpu,pmem,rss,etime,cmd --sort=-pcpu 2>&1 | head -80 || true

section "Top processes by RSS"
ps -eo pid,ppid,user,stat,psr,pcpu,pmem,rss,etime,cmd --sort=-rss 2>&1 | head -80 || true

section "Top snapshot"
top -b -n1 -w512 2>&1 | head -80 || true
