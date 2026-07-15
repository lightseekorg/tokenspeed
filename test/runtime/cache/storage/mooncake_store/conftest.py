import pytest

from tokenspeed.runtime.cache.kvstore_storage import KVStoreStorageConfig


def pytest_addoption(parser):
    parser.addoption(
        "--mooncake-config-path",
        help="Explicit Mooncake JSON config used by the live storage tests.",
    )


@pytest.fixture
def mooncake_storage_config(pytestconfig):
    config_path = pytestconfig.getoption("--mooncake-config-path")
    if not config_path:
        pytest.skip("live Mooncake test requires --mooncake-config-path")
    return KVStoreStorageConfig(
        is_mla_model=False,
        tp_rank=0,
        tp_size=1,
        model_name=None,
        is_page_first_layout=True,
        extra_config={"config_path": config_path},
    )
