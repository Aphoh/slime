import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
FULLY_ASYNC_PATH = REPO_ROOT / "slime" / "rollout" / "fully_async_rollout.py"


def _load_fully_async_module():
    stub_names = [
        "slime.rollout.sglang_rollout",
        "slime.utils.async_utils",
        "slime.utils.http_utils",
        "slime.utils.speedscope_trace",
        "slime.utils.types",
    ]
    saved_modules = {name: sys.modules.get(name) for name in stub_names}

    sglang_rollout_module = types.ModuleType("slime.rollout.sglang_rollout")
    sglang_rollout_module.GenerateState = object
    sglang_rollout_module.generate_and_rm = object
    sglang_rollout_module.generate_and_rm_group = object
    async_utils_module = types.ModuleType("slime.utils.async_utils")
    async_utils_module.run = lambda coro: coro
    http_utils_module = types.ModuleType("slime.utils.http_utils")
    http_utils_module.get_rollout_num_engines = lambda _args: 1
    trace_module = types.ModuleType("slime.utils.speedscope_trace")
    trace_module.trace_span = lambda *_args, **_kwargs: contextlib.nullcontext()
    types_module = types.ModuleType("slime.utils.types")
    types_module.Sample = object

    try:
        sys.modules["slime.rollout.sglang_rollout"] = sglang_rollout_module
        sys.modules["slime.utils.async_utils"] = async_utils_module
        sys.modules["slime.utils.http_utils"] = http_utils_module
        sys.modules["slime.utils.speedscope_trace"] = trace_module
        sys.modules["slime.utils.types"] = types_module

        spec = importlib.util.spec_from_file_location("fully_async_rollout_for_test", FULLY_ASYNC_PATH)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


def test_mock_trainer_timing_uses_token_count_and_update_interval():
    rollout = _load_fully_async_module()
    args = SimpleNamespace(
        mock_trainer_tokens_per_second=100.0,
        mock_weight_update_seconds=7.5,
        update_weights_interval=2,
    )
    data = [
        [SimpleNamespace(tokens=[1, 2, 3], response_length=99)],
        [SimpleNamespace(tokens=[], response_length=5)],
    ]

    assert rollout._mock_trainer_timing(args, 0, data) == (8, 0.08, 0.0)
    assert rollout._mock_trainer_timing(args, 1, data) == (8, 0.08, 7.5)


def test_async_worker_summary_is_structured_and_includes_shutdown_counters(monkeypatch, capsys):
    rollout = _load_fully_async_module()
    rollout.GenerateState = lambda _args: SimpleNamespace(sampling_params={})
    monkeypatch.setenv("SWEPRO_ASYNC_MAX_STARTED_GROUPS", "7")
    args = SimpleNamespace(
        rollout_batch_size=8,
        n_samples_per_prompt=1,
        sglang_server_concurrency=4,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )
    worker = rollout.AsyncRolloutWorker(args, SimpleNamespace())
    worker.started_count = 6
    worker.completed_count = 4
    worker.accepted_group_count = 2
    worker.returned_group_count = 1
    worker.shutdown_drained_count = 2
    worker.shutdown_cancelled_count = 3

    worker.log_summary("batch_returned", rollout_id=0, target_groups=2, returned_groups=2, duration_s=12.5)

    line = capsys.readouterr().out.strip()
    prefix, payload_text = line.split(" ", 1)
    payload = json.loads(payload_text)
    assert prefix == "SWEPRO_ASYNC_WORKER_SUMMARY"
    assert payload["event"] == "batch_returned"
    assert payload["started"] == 6
    assert payload["completed"] == 4
    assert payload["accepted"] == 2
    assert payload["returned_to_buffer"] == 1
    assert payload["shutdown_drained"] == 2
    assert payload["shutdown_cancelled"] == 3
    assert payload["max_started_groups"] == 7
    assert payload["inflight_by_count"] == 2
    assert payload["target_groups"] == 2
    assert payload["returned_groups"] == 2


def test_stop_global_worker_keeps_handle_when_thread_does_not_stop():
    rollout = _load_fully_async_module()

    class Worker:
        def stop(self):
            return False

    worker = Worker()
    rollout._global_worker = worker

    rollout.stop_global_worker()

    assert rollout._global_worker is worker

    rollout._global_worker = None
