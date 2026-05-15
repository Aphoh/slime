import sys
import types
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

nats_module = types.ModuleType("nats")
nats_aio_module = types.ModuleType("nats.aio")
nats_client_module = types.ModuleType("nats.aio.client")
nats_client_module.Client = object
sys.modules.setdefault("nats", nats_module)
sys.modules.setdefault("nats.aio", nats_aio_module)
sys.modules.setdefault("nats.aio.client", nats_client_module)

sglang_rollout_module = types.ModuleType("slime.rollout.sglang_rollout")
sglang_rollout_module.GenerateState = object
mask_utils_module = types.ModuleType("slime.utils.mask_utils")
mask_utils_module.MultiTurnLossMaskGenerator = object
types_module = types.ModuleType("slime.utils.types")
types_module.Sample = object
sys.modules.setdefault("slime.rollout.sglang_rollout", sglang_rollout_module)
sys.modules.setdefault("slime.utils.mask_utils", mask_utils_module)
sys.modules.setdefault("slime.utils.types", types_module)

import generate_with_swebench_pro  # noqa: E402


def test_turn_max_tokens_defaults_to_8192(monkeypatch):
    monkeypatch.delenv("SWEPRO_TURN_MAX_TOKENS", raising=False)

    assert generate_with_swebench_pro._turn_max_tokens(SimpleNamespace(rollout_max_response_len=131072)) == 8192


def test_turn_max_tokens_zero_disables_per_turn_cap(monkeypatch):
    monkeypatch.setenv("SWEPRO_TURN_MAX_TOKENS", "0")

    assert generate_with_swebench_pro._turn_max_tokens(SimpleNamespace(rollout_max_response_len=131072)) == 131072


def test_turn_max_tokens_unlimited_still_respects_sampling_and_rollout_limits(monkeypatch):
    monkeypatch.setenv("SWEPRO_TURN_MAX_TOKENS", "unlimited")

    assert (
        generate_with_swebench_pro._turn_max_tokens(
            SimpleNamespace(rollout_max_response_len=131072),
            {"max_new_tokens": 32768},
        )
        == 32768
    )
