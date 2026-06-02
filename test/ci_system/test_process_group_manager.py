import signal

import process_group_manager as pgm
from process_group_manager import ProcessGroupManager


def test_cleanup_stale_kills_all_persisted_process_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path)
    monkeypatch.setattr(pgm.time, "sleep", lambda _seconds: None)

    killed = []
    monkeypatch.setattr(
        pgm.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )

    pgm._add_pgid("gb200-runner", 111)
    pgm._add_pgid("gb200-runner", 222)

    manager = ProcessGroupManager("gb200-runner")
    manager.cleanup_stale()

    assert killed == [
        (111, signal.SIGTERM),
        (222, signal.SIGTERM),
        (111, signal.SIGKILL),
        (222, signal.SIGKILL),
    ]
    assert not pgm._pgid_path("gb200-runner").exists()


def test_terminate_all_kills_persisted_process_groups_without_tracked_parents(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(pgm, "_PGID_DIR", tmp_path)
    monkeypatch.setattr(pgm.time, "sleep", lambda _seconds: None)

    killed = []
    monkeypatch.setattr(
        pgm.os,
        "killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )

    pgm._add_pgid("gb200-runner", 111)
    pgm._add_pgid("gb200-runner", 222)

    manager = ProcessGroupManager("gb200-runner")
    manager.terminate_all()

    assert killed == [
        (111, signal.SIGTERM),
        (222, signal.SIGTERM),
        (111, signal.SIGKILL),
        (222, signal.SIGKILL),
    ]
    assert not pgm._pgid_path("gb200-runner").exists()
