"""How many CPUs this process may actually use.

``os.cpu_count()`` counts the host's cores, not the container's share of them.
On Railway an 8-vCPU service saw 48, so 48 page renderers and 48 torch threads
fought over 8 CPUs and embedding 208 chunks ran past ten minutes. The cgroup
quota is what the scheduler enforces, so it is read first.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _quota(v2: Path, v1_quota: Path, v1_period: Path) -> int | None:
    try:
        quota, period = v2.read_text().split()[:2]
    except (OSError, ValueError):
        try:
            quota, period = v1_quota.read_text().strip(), v1_period.read_text().strip()
        except OSError:
            return None
    if quota in ("max", "-1"):
        return None
    try:
        return max(1, math.ceil(int(quota) / int(period)))
    except (ValueError, ZeroDivisionError):
        return None


def available_cpus(*, v2: Path = CGROUP_V2, v1_quota: Path = CGROUP_V1_QUOTA,
                   v1_period: Path = CGROUP_V1_PERIOD) -> int:
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count() or 1
    quota = _quota(v2, v1_quota, v1_period)
    return min(allowed, quota) if quota else allowed
