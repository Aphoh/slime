import contextlib
import importlib.util
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
