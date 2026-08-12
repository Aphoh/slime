import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.backends.sglang_utils import external
from slime.backends.sglang_utils.external import (
    ExternalEngineInfo,
    apply_external_engine_info_to_args,
    discover_external_engines,
    external_engine_init_kwargs,
    start_external_rollout_servers,
)
from slime.utils.http_utils import get_rollout_num_engines

NUM_GPUS = 0


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_discover_external_engines_reads_server_info(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 30.0
        assert url == "http://host1:10090/server_info"
        return _Response(
            {
                "tp_size": 4,
                "pp_size": 2,
                "dp_size": 1,
                "ep_size": 4,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)

    infos = discover_external_engines(["host1:10090"])

    assert len(infos) == 1
    info = infos[0]
    assert info.url == "http://host1:10090"
    assert info.host == "host1"
    assert info.port == 10090
    assert info.worker_type == "regular"
    assert info.num_gpus == 8
    assert info.server_info["tp_size"] == 4
    assert info.server_info["pp_size"] == 2
    assert info.server_info["dp_size"] == 1
    assert info.server_info["ep_size"] == 4
    assert info.parallel_config == {"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 1}


def test_start_external_rollout_servers_exposes_parallel_configs(monkeypatch):
    class FakeActor:
        init = Namespace(remote=lambda **kwargs: kwargs)

    class FakeActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return FakeActor()

    ray = types.ModuleType("ray")
    ray.remote = lambda actor_class: FakeActorClass()
    sglang_engine = types.ModuleType("slime.backends.sglang_utils.sglang_engine")
    sglang_engine.SGLangEngine = object
    ray_utils = types.ModuleType("slime.ray.utils")
    ray_utils.add_default_ray_env_vars = lambda: {}
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "slime.backends.sglang_utils.sglang_engine", sglang_engine)
    monkeypatch.setitem(sys.modules, "slime.ray.utils", ray_utils)

    info = ExternalEngineInfo(
        url="http://host1:10090",
        host="host1",
        port=10090,
        worker_type="regular",
        num_gpus=8,
        server_info={"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 2},
    )
    args = Namespace(
        rollout_external_engine_infos=[info.to_dict()],
        rollout_external_dynamic_discovery_path=None,
        rollout_external_rollout_url=None,
    )

    servers, init_handles = start_external_rollout_servers(args, start_router=lambda *args, **kwargs: ("host1", 30000))

    assert servers["default"].engine_parallel_configs == [{"tp_size": 4, "pp_size": 2, "ep_size": 4, "moe_dp_size": 2}]
    assert len(init_handles) == 1


def test_discover_external_engines_uses_control_base_url(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 30.0
        assert url == "http://worker:9090/engine/server_info"
        return _Response({"tp_size": 2})

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)

    info = discover_external_engines(["worker:9090/engine"])[0]

    assert info.url == "http://worker:9090/engine"
    assert external_engine_init_kwargs(info)["control_url"] == "http://worker:9090/engine"


def test_apply_external_engine_info_uses_dynamic_discovery_control_base_urls(monkeypatch):
    monkeypatch.setattr(external, "load_function", lambda path: lambda args: ["worker:9090/engine"])

    def fake_get(url, timeout):
        assert url == "http://worker:9090/engine/server_info"
        return _Response({"tp_size": 2})

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external_engine_addrs=None,
        rollout_external_dynamic_discovery_path="example.discover",
        rollout_external_rollout_url="http://frontend:8000",
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external_engine_infos[0]["url"] == "http://worker:9090/engine"
    assert args.rollout_external_engine_infos[0]["host"] == "worker"
    assert args.rollout_external_engine_infos[0]["port"] == 9090


def test_dynamic_discovery_refreshes_changed_external_engine_membership(monkeypatch):
    args = Namespace(
        rollout_external_engine_addrs=None,
        rollout_external_dynamic_discovery_path="example.discover",
        rollout_external_rollout_url="http://frontend:8000",
    )
    old_info = external.ExternalEngineInfo(
        "http://old:9090/engine", "old", 9090, "regular", 2, server_info={"tp_size": 2}
    )
    new_info = external.ExternalEngineInfo(
        "http://new:9090/engine", "new", 9090, "regular", 4, server_info={"tp_size": 4}
    )
    server = external.ExternalRolloutServer(
        engines=["old-engine"],
        engine_gpu_counts=[2],
        engine_gpu_offsets=[0],
        args=args,
        engine_infos=[old_info],
        register_to_router=False,
    )
    monkeypatch.setattr(external, "discover_external_engine_infos", lambda _args: [new_info])
    monkeypatch.setattr(
        external,
        "_start_external_engine_actors",
        lambda *args, **kwargs: (["new-engine"], []),
    )

    assert server.refresh() is True
    assert server.engines == ["new-engine"]
    assert server.engine_gpu_counts == [4]
    assert server.engine_gpu_offsets == [0]
    assert server.num_new_engines == 1
    assert server.retired_engines == ["old-engine"]
    assert args.rollout_external_engine_infos == [new_info.to_dict()]


def test_dynamic_discovery_skips_unchanged_external_engine_membership(monkeypatch):
    args = Namespace(
        rollout_external_engine_addrs=None,
        rollout_external_dynamic_discovery_path="example.discover",
        rollout_external_rollout_url="http://frontend:8000",
    )
    info = external.ExternalEngineInfo(
        "http://worker:9090/engine", "worker", 9090, "regular", 2, server_info={"tp_size": 2}
    )
    server = external.ExternalRolloutServer(
        engines=["engine"],
        engine_gpu_counts=[2],
        engine_gpu_offsets=[0],
        args=args,
        engine_infos=[info],
        register_to_router=False,
    )
    monkeypatch.setattr(external, "discover_external_engine_infos", lambda _args: [info])
    monkeypatch.setattr(
        external,
        "_start_external_engine_actors",
        lambda *args, **kwargs: pytest.fail("unchanged membership must not recreate control actors"),
    )

    assert server.refresh() is False
    assert server.engines == ["engine"]


def test_apply_external_engine_info_handles_pd(monkeypatch):
    payloads = {
        "http://prefill:10090/server_info": {
            "tp_size": 2,
            "pp_size": 1,
            "dp_size": 1,
            "ep_size": 1,
            "disaggregation_mode": "prefill",
            "disaggregation_bootstrap_port": 12090,
        },
        "http://decode:10091/server_info": {
            "tp_size": 4,
            "pp_size": 1,
            "dp_size": 2,
            "ep_size": 2,
            "disaggregation_mode": "decode",
        },
    }

    def fake_get(url, timeout):
        return _Response(payloads[url])

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["prefill:10090", "decode:10091"],
        rollout_external_dynamic_discovery_path=None,
        rollout_external_rollout_url=None,
        rollout_num_gpus=None,
        rollout_num_gpus_per_engine=1,
        sglang_pipeline_parallel_size=1,
        sglang_data_parallel_size=1,
        sglang_expert_parallel_size=1,
        sglang_enable_dp_attention=False,
        router_pd_disaggregation=False,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is False
    assert args.rollout_num_gpus == 6
    assert args.rollout_num_engines == 2
    assert get_rollout_num_engines(args) == 2
    assert [info["worker_type"] for info in args.rollout_external_engine_infos] == ["prefill", "decode"]
    assert [info["num_gpus"] for info in args.rollout_external_engine_infos] == [2, 4]
    assert [info["server_info"]["dp_size"] for info in args.rollout_external_engine_infos] == [1, 2]
    assert args.rollout_external_engine_infos[0]["disaggregation_bootstrap_port"] == 12090


def test_apply_external_engine_info_preserves_router_pd_flag(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://regular:10090/server_info"
        return _Response(
            {
                "tp_size": 2,
                "pp_size": 1,
                "disaggregation_mode": "null",
            }
        )

    monkeypatch.setattr("slime.backends.sglang_utils.external.requests.get", fake_get)
    args = Namespace(
        rollout_external=True,
        rollout_external_engine_addrs=["regular:10090"],
        rollout_external_dynamic_discovery_path=None,
        rollout_external_rollout_url=None,
        router_pd_disaggregation=True,
    )

    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is True
    assert args.rollout_num_gpus == 2
    assert args.rollout_num_engines == 1


def test_apply_external_engine_info_requires_addrs_or_discovery():
    args = Namespace(rollout_external_engine_addrs=None, rollout_external_dynamic_discovery_path=None)

    with pytest.raises(ValueError, match="rollout-external-engine-addrs or"):
        apply_external_engine_info_to_args(args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
