"""Phase 1: WeightTransferConfig + server args + gating."""

import argparse

import pytest

from tokenspeed.runtime.engine.weight_transfer.config import (
    SUPPORTED_BACKENDS,
    WeightTransferConfig,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


class TestWeightTransferConfig:
    def test_default_backend_is_nccl(self):
        assert WeightTransferConfig().backend == "nccl"

    def test_supported_backends(self):
        assert SUPPORTED_BACKENDS == ("nccl", "ipc")

    @pytest.mark.parametrize("backend", ["nccl", "ipc"])
    def test_valid_backends(self, backend):
        assert WeightTransferConfig(backend=backend).backend == backend

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported weight transfer backend"):
            WeightTransferConfig(backend="rdma")

    def test_from_json_none_and_empty_default(self):
        assert WeightTransferConfig.from_json(None).backend == "nccl"
        assert WeightTransferConfig.from_json("").backend == "nccl"

    def test_from_json_roundtrip(self):
        assert WeightTransferConfig.from_json('{"backend":"ipc"}').backend == "ipc"

    def test_from_json_bad_json_raises(self):
        with pytest.raises(ValueError, match="valid JSON"):
            WeightTransferConfig.from_json("not json")

    def test_from_json_non_object_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            WeightTransferConfig.from_json("[1, 2]")

    def test_from_json_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown weight-transfer-config keys"):
            WeightTransferConfig.from_json('{"backend":"nccl","foo":1}')

    def test_from_json_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported weight transfer backend"):
            WeightTransferConfig.from_json('{"backend":"bogus"}')


class TestServerArgsConfig:
    # The control plane is ungated (always on); no enable flag.
    def test_no_enable_flag_attribute(self):
        sa = ServerArgs(model="m")
        assert not hasattr(sa, "enable_weight_transfer")
        assert not hasattr(sa, "weight_transfer_enabled")

    def test_get_weight_transfer_config_default(self):
        sa = ServerArgs(model="m")
        assert sa.get_weight_transfer_config().backend == "nccl"

    def test_get_weight_transfer_config_from_json(self):
        sa = ServerArgs(model="m", weight_transfer_config='{"backend":"ipc"}')
        assert sa.get_weight_transfer_config().backend == "ipc"


class TestServerArgsCli:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return ServerArgs.from_cli_args(parser.parse_args(argv))

    def test_cli_config(self):
        sa = self._parse(
            ["--model", "m", "--weight-transfer-config", '{"backend":"nccl"}']
        )
        assert sa.get_weight_transfer_config().backend == "nccl"

    def test_cli_default_no_config(self):
        sa = self._parse(["--model", "m"])
        assert sa.weight_transfer_config is None

    def test_no_enable_flag_registered(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        opts = {o for a in parser._actions for o in a.option_strings}
        assert "--enable-weight-transfer" not in opts
        assert "--no-enable-weight-transfer" not in opts
