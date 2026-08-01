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

"""Unit tests for launcher-environment topology discovery."""

import pytest

from tokenspeed.runtime.utils import launcher

HEAD_IP = "10.40.1.228"


@pytest.mark.parametrize(
    "hostlist,expected",
    [
        ("cn10", "cn10"),
        ("cn10,cn15", "cn10"),
        ("inkwell-iron-cn[15-16]", "inkwell-iron-cn15"),
        ("cn[10,15]", "cn10"),
        ("cn[01-03,07]", "cn01"),
        ("cn[07,01-03]", "cn07"),
        ("cn[01-02],dgx[1-2]", "cn01"),
        ("rack[1-2]node[3-4]", "rack1node3"),
        ("cn[001-100]-ib", "cn001-ib"),
        ("  cn[5-9]  ", "cn5"),
    ],
)
def test_first_host(hostlist, expected):
    assert launcher.first_host(hostlist) == expected


def test_first_host_rejects_empty():
    with pytest.raises(ValueError, match="empty hostlist"):
        launcher.first_host("")


def _multi_node_env(**overrides):
    env = {
        launcher.SLURM_STEP_NUM_NODES: "2",
        launcher.SLURM_NODEID: "1",
        launcher.SLURM_STEP_NODELIST: "inkwell-iron-cn[15-16]",
    }
    env.update(overrides)
    return env


@pytest.fixture
def resolvable_head(monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostbyname", lambda host: HEAD_IP)
    monkeypatch.setattr(launcher, "local_ipv4_addresses", lambda: {HEAD_IP})


def test_no_launcher_environment_returns_none():
    assert launcher.detect_topology({}) is None


def test_single_node_step_returns_none():
    assert launcher.detect_topology(_multi_node_env(SLURM_STEP_NUM_NODES="1")) is None


def test_batch_script_environment_is_not_a_multi_node_step():
    """A bare `ts serve` in a multi-node sbatch is a deliberate single-node run."""
    assert (
        launcher.detect_topology(
            {
                "SLURM_NNODES": "2",
                launcher.SLURM_NODEID: "0",
                "SLURM_JOB_NODELIST": "inkwell-iron-cn[02-03]",
            }
        )
        is None
    )


def test_detects_multi_node_step(resolvable_head):
    topology = launcher.detect_topology(_multi_node_env())
    assert (topology.nnodes, topology.node_rank) == (2, 1)
    assert topology.head_host == HEAD_IP
    assert topology.source == "Slurm"
    assert topology.dist_init_addr == f"{HEAD_IP}:{launcher.DIST_INIT_DEFAULT_PORT}"


def test_dist_init_port_is_a_constant(resolvable_head):
    """Every node must reach the same port without coordinating."""
    ports = {
        launcher.detect_topology(_multi_node_env(SLURM_NODEID=rank)).dist_init_port
        for rank in ("0", "1")
    }
    assert ports == {launcher.DIST_INIT_DEFAULT_PORT}


def test_dist_init_port_is_outside_the_ephemeral_range():
    """A port the kernel hands out to others is not usable for a rendezvous."""
    launcher.check_dist_init_port(launcher.DIST_INIT_DEFAULT_PORT)


def test_ephemeral_dist_init_port_is_rejected(monkeypatch):
    monkeypatch.setattr(launcher, "_ephemeral_port_range", lambda: (32768, 60999))
    with pytest.raises(ValueError, match="ephemeral port range"):
        launcher.check_dist_init_port(60486)


def test_port_check_is_skipped_without_a_known_range(monkeypatch):
    monkeypatch.setattr(launcher, "_ephemeral_port_range", lambda: None)
    launcher.check_dist_init_port(60486)


def test_detection_does_not_check_the_rendezvous_port(monkeypatch, resolvable_head):
    """The port only has to be usable where the derived address is adopted."""
    monkeypatch.setattr(launcher, "_ephemeral_port_range", lambda: (1024, 60999))
    topology = launcher.detect_topology(_multi_node_env())
    assert topology.dist_init_port == launcher.DIST_INIT_DEFAULT_PORT


def test_missing_node_id_is_an_error():
    env = _multi_node_env()
    del env[launcher.SLURM_NODEID]
    with pytest.raises(ValueError, match="SLURM_NODEID is unset"):
        launcher.detect_topology(env)


def test_missing_nodelist_is_an_error():
    env = _multi_node_env()
    del env[launcher.SLURM_STEP_NODELIST]
    with pytest.raises(ValueError, match="SLURM_STEP_NODELIST is unset"):
        launcher.detect_topology(env)


def test_non_integer_nnodes_is_an_error():
    with pytest.raises(ValueError, match="not an integer"):
        launcher.detect_topology(_multi_node_env(SLURM_STEP_NUM_NODES="two"))


def test_node_id_out_of_range_is_an_error():
    with pytest.raises(ValueError, match="out of range"):
        launcher.detect_topology(_multi_node_env(SLURM_NODEID="2"))


def test_unresolvable_head_is_an_error(monkeypatch):
    def boom(host):
        raise OSError("no such host")

    monkeypatch.setattr(launcher.socket, "gethostbyname", boom)
    with pytest.raises(ValueError, match="cannot resolve the head node"):
        launcher.detect_topology(_multi_node_env())


def test_loopback_head_on_follower_is_an_error(monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostbyname", lambda host: "127.0.1.1")
    with pytest.raises(ValueError, match="resolves to loopback"):
        launcher.detect_topology(_multi_node_env(SLURM_NODEID="1"))


def test_loopback_head_on_rank_zero_uses_local_address(monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostbyname", lambda host: "127.0.1.1")
    monkeypatch.setattr(launcher, "_interface_ipv4", lambda iface: HEAD_IP)
    monkeypatch.setattr(launcher, "local_ipv4_addresses", lambda: {HEAD_IP})
    topology = launcher.detect_topology(
        _multi_node_env(SLURM_NODEID="0", NCCL_SOCKET_IFNAME="enP6p9s0np0")
    )
    assert topology.head_host == HEAD_IP


def test_rank_zero_rejects_a_non_local_head(monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostbyname", lambda host: HEAD_IP)
    monkeypatch.setattr(launcher, "local_ipv4_addresses", lambda: {"10.40.1.235"})
    with pytest.raises(ValueError, match="not bound to any local interface"):
        launcher.detect_topology(_multi_node_env(SLURM_NODEID="0"))


def test_follower_accepts_a_non_local_head(monkeypatch):
    monkeypatch.setattr(launcher.socket, "gethostbyname", lambda host: HEAD_IP)
    monkeypatch.setattr(launcher, "local_ipv4_addresses", lambda: {"10.40.1.235"})
    assert launcher.detect_topology(_multi_node_env()).head_host == HEAD_IP
