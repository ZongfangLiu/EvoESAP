#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU availability checker across local + SSH servers.

- Step 1: Shows how many GPUs are FREE on each server and ranks servers.
- Step 2: Shows detailed GPU stats per server (ranked by free memory).

A GPU is considered FREE if:
  - It has NO active compute processes, AND
  - memory.used <= --free-mem-threshold (default: 500 MiB)

Notes:
- Alias "8_5090_28G" is treated as LOCAL (commands run without SSH).
- Other aliases are executed via SSH. Ensure `ssh <host>` is passwordless.
"""

import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

# ---- Configure your SSH targets (matching your ~/.ssh/config) ----
SERVERS = [
    "8_5090_28G",         # LOCAL
    "8_4090_48G",
    "8_4090_24G_20022",
    "8_4090_24G_20021",
    "8_L40S_48G",
]

# ---- Which aliases are local (no SSH) ----
# LOCAL_ALIASES: Set[str] = {"8_5090_28G"}
LOCAL_ALIASES: Set[str] = {}

SSH_TIMEOUT = 20  # seconds


@dataclass
class GPUInfo:
    index: int
    uuid: str
    name: str
    mem_total: int
    mem_used: int
    mem_free: int
    has_process: bool  # True if any compute process is running on this GPU
    users: List[str]   # usernames of processes using this GPU (unique, sorted)

    @property
    def is_free(self) -> bool:
        return (not self.has_process) and (self.mem_used <= ARGS.free_mem_threshold)


def run_cmd(server: str, remote_cmd: str) -> Tuple[int, str, str]:
    """
    Run a shell command either locally (if server in LOCAL_ALIASES) or via SSH.
    Returns (rc, stdout, stderr).
    """
    try:
        if server in LOCAL_ALIASES:
            # Run locally using bash -lc to allow pipes/redirs
            res = subprocess.run(
                ["bash", "-lc", remote_cmd],
                capture_output=True,
                text=True,
                timeout=SSH_TIMEOUT,
            )
            return res.returncode, res.stdout, res.stderr
        else:
            # Run via SSH
            res = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", server, remote_cmd],
                capture_output=True,
                text=True,
                timeout=SSH_TIMEOUT,
            )
            return res.returncode, res.stdout, res.stderr
    except Exception as e:
        return 255, "", str(e)


def _fetch_pid_user_map(server: str) -> Dict[int, str]:
    """
    Return a mapping of PID -> username on the target host.
    Uses: ps -eo pid,user --no-headers
    """
    rc, out, _ = run_cmd(server, "ps -eo pid,user --no-headers || true")
    if rc != 0:
        return {}
    mapping: Dict[int, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_s, user = line.split(maxsplit=1)
            mapping[int(pid_s)] = user.strip()
        except Exception:
            continue
    return mapping


def fetch_gpu_table(server: str) -> List[GPUInfo]:
    """
    Query nvidia-smi for:
      - GPU list: index, uuid, name, total/used/free (MiB)
      - Active compute apps: (gpu_uuid, pid) -> users via ps
    Returns list of GPUInfo.
    """
    # 1) Active compute apps: gpu_uuid,pid
    rc_p, out_p, _ = run_cmd(
        server,
        "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true",
    )
    uuid_to_pids: Dict[str, Set[int]] = {}
    if rc_p == 0:
        for line in out_p.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            gpu_uuid, pid_s = parts[:2]
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            uuid_to_pids.setdefault(gpu_uuid, set()).add(pid)

    # 2) PID -> user (one shot)
    pid_user = _fetch_pid_user_map(server)

    # 3) Base GPU info
    rc_g, out_g, err_g = run_cmd(
        server,
        "nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free "
        "--format=csv,noheader,nounits",
    )
    if rc_g != 0:
        raise RuntimeError(f"nvidia-smi failed on {server}: {err_g.strip() or 'Unknown error'}")

    infos: List[GPUInfo] = []
    for line in out_g.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx_s, uuid, name, total_s, used_s, free_s = parts[:6]
        try:
            pids = uuid_to_pids.get(uuid, set())
            users = sorted({pid_user.get(pid, "unknown") for pid in pids})
            infos.append(
                GPUInfo(
                    index=int(idx_s),
                    uuid=uuid,
                    name=name,
                    mem_total=int(total_s),
                    mem_used=int(used_s),
                    mem_free=int(free_s),
                    has_process=(len(pids) > 0),
                    users=users,
                )
            )
        except ValueError:
            continue

    return infos


def print_summary_ranked(server_gpus: Dict[str, List[GPUInfo]]) -> None:
    """Step 1: Show how many FREE GPUs each server has, ranked, plus free indices and memory stats."""
    summary = []
    free_indices_map = {}
    free_mem_map = {}
    for server, gpus in server_gpus.items():
        free_idxs = [g.index for g in gpus if g.is_free]
        free_count = len(free_idxs)
        total = len(gpus)
        free_mem_total = sum(g.mem_free for g in gpus if g.is_free)  # only free GPUs
        free_mem_avg = (free_mem_total / free_count) if free_count > 0 else 0
        summary.append((server, free_count, total, free_mem_total, free_mem_avg))
        free_indices_map[server] = free_idxs
        free_mem_map[server] = (free_mem_total, free_mem_avg)

    # Rank by: free_count desc, then avg free mem desc, then server name
    summary.sort(key=lambda x: (-x[1], -x[4], x[0]))

    print("\n==================== SUMMARY (RANKED) ====================")
    print(f"Criteria for FREE: no processes AND mem_used <= {ARGS.free_mem_threshold} MiB\n")
    # Header
    print(f"{'Server':<24} {'Free':>7} {'Total':>7} {'FreeMem(GB)':>14} {'AvgFree(GB)':>14}   FreeIdx")
    print("-" * 80)
    # Rows
    for server, free_count, total, free_mem_total, free_mem_avg in summary:
        tag = " (local)" if server in LOCAL_ALIASES else ""
        free_idxs = free_indices_map[server]
        idx_str = "[" + ",".join(map(str, free_idxs)) + "]" if free_idxs else "[]"
        free_mem_gb = free_mem_total / 1024
        free_avg_gb = free_mem_avg / 1024
        print(f"{server+tag:<24} {free_count:>3}/{total:<3} {total:>7} {free_mem_gb:>14.1f} {free_avg_gb:>14.1f}   {idx_str}")
    print("=" * 80 + "\n")


def print_details_ranked(server_gpus: Dict[str, List[GPUInfo]]) -> None:
    """Step 2: Print per-server GPU details ranked by free memory."""
    for server in sorted(server_gpus.keys()):
        gpus = server_gpus[server]
        tag = " (local)" if server in LOCAL_ALIASES else ""
        if not gpus:
            print(f"----- {server}{tag} -----\n(No GPUs detected)\n")
            continue

        # Sort by: FREE first, then free memory desc, then index
        def sort_key(g: GPUInfo):
            return (0 if g.is_free else 1, -g.mem_free, g.index)

        gpus_sorted = sorted(gpus, key=sort_key)

        print(f"-------------------- {server}{tag} --------------------")
        print(f"{'Idx':>3}  {'Model':<24}  {'Used/Total(MiB)':>17}  {'Free':>6}  {'Proc':>5}  {'Status':>6}  {'Users':<}")
        for g in gpus_sorted:
            status = "FREE" if g.is_free else "BUSY"
            proc = "yes" if g.has_process else "no"
            users_str = ",".join(g.users) if (g.users and not g.is_free) else ""
            print(f"{g.index:>3}  {g.name:<24}  {g.mem_used:>5}/{g.mem_total:<10}  {g.mem_free:>6}  {proc:>5}  {status:>6}  {users_str}")
        print()  # blank line after each server


def main():
    server_gpus: Dict[str, List[GPUInfo]] = {}
    for server in SERVERS:
        print(f"Checking {server}{' (local)' if server in LOCAL_ALIASES else ''} ...")
        try:
            gpus = fetch_gpu_table(server)
            server_gpus[server] = gpus
        except Exception as e:
            print(f"  Error on {server}: {e}")
            server_gpus[server] = []

    print_details_ranked(server_gpus)
    print_summary_ranked(server_gpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check GPU availability across local + SSH servers.")
    parser.add_argument(
        "--free-mem-threshold",
        type=int,
        default=500,
        help="Max used memory (MiB) to still consider a GPU FREE (default: 500).",
    )
    ARGS = parser.parse_args()
    main()