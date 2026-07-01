import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

nats_module = types.ModuleType("nats")
nats_aio_module = types.ModuleType("nats.aio")
nats_client_module = types.ModuleType("nats.aio.client")
nats_client_module.Client = object
sys.modules.setdefault("nats", nats_module)
sys.modules.setdefault("nats.aio", nats_aio_module)
sys.modules.setdefault("nats.aio.client", nats_client_module)


class _SampleStub:
    class Status:
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    def __init__(self, *, prompt="", label=None, metadata=None):
        self.prompt = prompt
        self.label = label
        self.metadata = metadata or {}
        self.index = 0
        self.session_id = None
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.loss_mask = None
        self.rollout_log_probs = None
        self.status = self.Status.PENDING
        self.reward = None
        self.remove_sample = False


sglang_rollout_module = types.ModuleType("slime.rollout.sglang_rollout")
sglang_rollout_module.GenerateState = object
mask_utils_module = types.ModuleType("slime.utils.mask_utils")
mask_utils_module.MultiTurnLossMaskGenerator = object
types_module = types.ModuleType("slime.utils.types")
types_module.Sample = _SampleStub
sys.modules.setdefault("slime.rollout.sglang_rollout", sglang_rollout_module)
sys.modules.setdefault("slime.utils.mask_utils", mask_utils_module)
sys.modules.setdefault("slime.utils.types", types_module)

import generate_with_swebench_pro  # noqa: E402

for module_name, stub in (
    ("generate_with_swebench_pro", generate_with_swebench_pro),
    ("nats", nats_module),
    ("nats.aio", nats_aio_module),
    ("nats.aio.client", nats_client_module),
    ("slime.rollout.sglang_rollout", sglang_rollout_module),
    ("slime.utils.mask_utils", mask_utils_module),
    ("slime.utils.types", types_module),
):
    if sys.modules.get(module_name) is stub:
        sys.modules.pop(module_name)


def test_swepro_model_uses_cli_dynamo_configuration(monkeypatch):
    for name in (
        "SWEPRO_DYNAMO_FRONTEND_URL",
        "DYNAMO_FRONTEND_URL",
        "SWEPRO_DYNAMO_API_MODE",
        "SWEPRO_MODEL",
        "SWEPRO_DYNAMO_METADATA_UPLOAD_URL",
        "SWEPRO_DYNAMO_METADATA_UPLOAD_FORMAT",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(generate_with_swebench_pro, "_MODEL", None)
    monkeypatch.setattr(
        generate_with_swebench_pro,
        "DirectCompletionsModel",
        lambda config: config,
    )
    args = SimpleNamespace(
        dynamo_frontend_url="https://dynamo.example/proxy",
        dynamo_api_mode="responses",
        dynamo_served_model_name="served-model",
        dynamo_metadata_upload_url="s3://rollouts/metadata",
        dynamo_metadata_upload_format="msgpack",
        hf_checkpoint="checkpoint-name",
        rollout_max_response_len=4096,
        rollout_temperature=0.7,
        rollout_top_p=0.9,
        rollout_top_k=20,
    )

    config = generate_with_swebench_pro._get_model(args)

    assert config.base_url == "https://dynamo.example/proxy"
    assert config.api_mode == "responses"
    assert config.model == "served-model"
    assert config.metadata_upload_url == "s3://rollouts/metadata"


def test_missing_sweagent_config_reports_required_root(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_with_swebench_pro, "_SWEAGENT_TEMPLATES", None)
    monkeypatch.setenv("SWE_AGENT_CONFIG_ROOT", str(tmp_path / "missing"))

    with pytest.raises(FileNotFoundError, match="SWE_AGENT_CONFIG_ROOT"):
        generate_with_swebench_pro._load_sweagent_templates()


def test_missing_image_name_requires_explicit_registry_owner(monkeypatch, tmp_path):
    (tmp_path / "helper_code").mkdir()
    monkeypatch.setenv("SWEPRO_EVAL_ROOT", str(tmp_path))
    monkeypatch.delenv("SWEPRO_DOCKERHUB_USERNAME", raising=False)

    with pytest.raises(ValueError, match="SWEPRO_DOCKERHUB_USERNAME"):
        generate_with_swebench_pro._metadata_image_name({"instance_id": "case-1", "repo": "org/repo"})


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


def test_tool_call_stop_is_reinserted_into_assistant_history_text():
    content = "thinking<tool_call><function=bash></function>"
    tool_calls = [{"function": {"name": "bash"}}]

    assert generate_with_swebench_pro._assistant_content_for_history(content, tool_calls).endswith("</tool_call>")


def test_tool_call_stop_is_not_duplicated_when_backend_returns_it():
    content = "thinking<tool_call><function=bash></function></tool_call>"
    tool_calls = [{"function": {"name": "bash"}}]

    assert generate_with_swebench_pro._assistant_content_for_history(content, tool_calls).count("</tool_call>") == 1


def test_submission_trace_fields_include_patch_file_preview():
    patch = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n"
        "-a\n"
        "+b\n"
    )

    fields = generate_with_swebench_pro._submission_trace_fields(patch)

    assert fields["patch_chars"] == len(patch)
    assert fields["patch_starts_with_diff"] is True
    assert fields["patch_file_count"] == 1
    assert fields["patch_files_preview"] == ["src/a.py"]
    assert len(fields["patch_sha256"]) == 16


def test_submission_trace_fields_prefers_session_worker_diagnostics():
    fields = generate_with_swebench_pro._submission_trace_fields(
        "",
        {"patch_diagnostics": {"patch_chars": 12, "submission_marker_count": 2}},
    )

    assert fields["patch_chars"] == 12
    assert fields["submission_marker_count"] == 2


def test_sweagent_session_stops_and_fails_on_content_filter(monkeypatch):
    class _Tokenizer:
        eos_token_id = 0
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "blocked"

    class _Model:
        tokenizer = _Tokenizer()
        tool_token_ids = SimpleNamespace(tool_close=154844)
        calls = 0

        @staticmethod
        def encode_prompt(messages, tools=None):
            return [1, 2]

        @classmethod
        def complete_prompt_ids(cls, *_args, **_kwargs):
            cls.calls += 1
            return {
                "content": "blocked",
                "extra": {
                    "generated_token_ids": [3],
                    "token_logprobs": [-0.1],
                    "finish_reason": "content_filter",
                    "stop_reason": "content_filter",
                    "response_tool_calls": [],
                },
            }

        @staticmethod
        def delete_response(_response_id):
            return None

    class _Client:
        step_calls = 0

        async def start(self, **_kwargs):
            return {"session_id": "session-1", "tools": []}

        async def step(self, *_args, **_kwargs):
            self.step_calls += 1
            raise AssertionError("content-filtered output must not enter the tool loop")

        async def submit(self, _session_id):
            return {"submission": ""}

        async def close(self, _session_id):
            return None

    client = _Client()
    monkeypatch.setattr(generate_with_swebench_pro, "_get_model", lambda _args: _Model())
    monkeypatch.setattr(generate_with_swebench_pro, "_get_trace_replay_store", lambda: None)
    monkeypatch.setattr(
        generate_with_swebench_pro,
        "_load_sweagent_templates",
        lambda: {
            "system_template": "system {{working_dir}}",
            "instance_template": "instance {{problem_statement}}",
            "next_step_template": "{{observation}}",
            "next_step_no_output_template": "next",
        },
    )
    monkeypatch.setattr(generate_with_swebench_pro, "build_agent_context_for_sample", lambda *_args: {})
    monkeypatch.setattr(generate_with_swebench_pro, "derive_tool_events_zmq_endpoint", lambda *_args: None)
    monkeypatch.setattr(
        generate_with_swebench_pro,
        "SweAgentSessionClient",
        SimpleNamespace(from_env=lambda _args: client),
    )
    monkeypatch.setattr(
        generate_with_swebench_pro,
        "_parse_tool_call_from_generated_tokens",
        lambda _model, content, _ids, _stops: (content, [], False),
    )
    sample = _SampleStub(
        prompt="fix it",
        label="instance-1",
        metadata={"instance_id": "instance-1", "base_commit": "abc", "image_name": "image"},
    )
    args = SimpleNamespace(
        partial_rollout=False,
        rollout_max_response_len=8,
        rollout_max_context_len=64,
        swepro_max_tool_calls=2,
        rollout_temperature=0.0,
        rollout_top_p=1.0,
        rollout_top_k=None,
    )

    result = asyncio.run(generate_with_swebench_pro._generate_sweagent_session(args, sample, {"max_new_tokens": 8}))

    assert _Model.calls == 1
    assert client.step_calls == 0
    assert result.status == generate_with_swebench_pro.Sample.Status.FAILED
    assert result.metadata["dynamo_finish_reason"] == "content_filter"
    assert result.response_length == 1
    assert result.rollout_log_probs == [-0.1]


NUM_GPUS = 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
