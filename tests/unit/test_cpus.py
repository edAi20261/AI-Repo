from __future__ import annotations

import os

from shared.cpus import available_cpus


def _limits(tmp_path, v2=None, quota=None, period=None):
    tmp_path = tmp_path / str(len(list(tmp_path.iterdir())))
    tmp_path.mkdir()
    paths = {"v2": tmp_path / "cpu.max", "v1_quota": tmp_path / "quota",
             "v1_period": tmp_path / "period"}
    if v2 is not None:
        paths["v2"].write_text(v2)
    if quota is not None:
        paths["v1_quota"].write_text(quota)
        paths["v1_period"].write_text(period)
    return paths


def test_a_container_limited_to_eight_cpus_uses_eight_not_the_hosts_count(tmp_path, monkeypatch):
    """An 8-vCPU Railway service saw the host's 48 cores, started 48 renderers and
    48 torch threads, and took over ten minutes to embed 208 chunks."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    assert available_cpus(**_limits(tmp_path, v2="800000 100000\n")) == 8
    assert available_cpus(**_limits(tmp_path, quota="250000", period="100000")) == 3


def test_no_quota_falls_back_to_the_cpus_the_process_may_run_on(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    assert available_cpus(**_limits(tmp_path, v2="max 100000\n")) == 6
    assert available_cpus(**_limits(tmp_path)) == 6
