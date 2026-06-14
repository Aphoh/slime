import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from slime.rollout.dynamo_client import (
    DynamoGeneration,
    apply_uploaded_metadata_sequence_to_sample,
    apply_uploaded_metadata_to_sample,
    build_dynamo_payload,
    build_metadata_upload_url,
)
from slime.utils.types import Sample

NUM_GPUS = 0


def test_build_dynamo_payload_supports_completions_and_responses():
    completions = build_dynamo_payload(
        api_mode="completions",
        model="model",
        prompt_token_ids=[1, 2],
        max_tokens=3,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        stop=["END"],
        stop_token_ids=[99],
        stream=True,
        return_logprobs=True,
        metadata_upload_url="s3://bucket/run/request",
    )
    assert completions["prompt"] == [1, 2]
    assert completions["stream"] is True
    assert completions["logprobs"] == 0
    assert completions["return_tokens_as_token_ids"] is True
    assert completions["stop"] == ["END"]
    assert completions["stop_token_ids"] == [99]
    assert completions["nvext"]["extra_fields"] == ["stop_reason", "completion_token_ids"]
    assert completions["nvext"]["metadata_upload"]["url"] == "s3://bucket/run/request"

    responses = build_dynamo_payload(
        api_mode="responses",
        model="model",
        prompt_token_ids=[1, 2],
        response_input=[{"role": "user", "content": "hello"}],
        previous_response_id="resp_previous",
        store=True,
        max_tokens=3,
        temperature=0.7,
        top_p=0.9,
        stop=["END"],
        stop_token_ids=[99],
        seed=123,
        skip_special_tokens=False,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
        stream=True,
        return_logprobs=True,
    )
    assert responses["input"] == [{"role": "user", "content": "hello"}]
    assert responses["previous_response_id"] == "resp_previous"
    assert responses["store"] is True
    assert responses["stop"] == ["END"]
    assert responses["stop_token_ids"] == [99]
    assert responses["nvext"]["token_data"] == [1, 2]
    assert responses["seed"] == 123
    assert responses["skip_special_tokens"] is False
    assert responses["no_stop_trim"] is True
    assert responses["include_stop_str_in_output"] is True
    assert responses["spaces_between_special_tokens"] is False
    assert responses["nvext"]["extra_fields"] == [
        "stop_reason",
        "completion_token_ids",
        "completion_token_logprobs",
    ]


def test_metadata_upload_url_is_stable_and_request_specific():
    first = build_metadata_upload_url("s3://bucket/run", "trajectory:1:try:0")
    repeated = build_metadata_upload_url("s3://bucket/run", "trajectory:1:try:0")
    second = build_metadata_upload_url("s3://bucket/run", "trajectory:1:try:1")

    assert first == repeated
    assert first != second
    assert first.startswith("s3://bucket/run/trajectory-1-try-0-")


def test_completions_stream_accumulates_exact_tokens_and_logprobs():
    generation = DynamoGeneration(api_mode="completions")
    generation.consume_sse(
        None,
        {
            "choices": [
                {
                    "text": "a",
                    "finish_reason": None,
                    "logprobs": {"tokens": ["token_id:11"], "token_logprobs": [-0.1]},
                }
            ],
            "nvext": {"completion_token_ids": [11]},
        },
    )
    generation.consume_sse(
        None,
        {
            "choices": [
                {
                    "text": "b",
                    "finish_reason": "stop",
                    "logprobs": {"tokens": ["token_id:12"], "token_logprobs": [-0.2]},
                }
            ],
            "nvext": {"completion_token_ids": [12], "stop_reason": "eos"},
        },
    )

    assert generation.token_ids == [11, 12]
    assert generation.token_logprobs == [-0.1, -0.2]
    assert generation.text == "ab"
    assert generation.finish_reason == "stop"
    assert generation.stop_reason == "eos"
    assert generation.terminal_event_received is True


def test_completions_stream_does_not_duplicate_nvext_logprobs():
    generation = DynamoGeneration(api_mode="completions")
    generation.consume_sse(
        None,
        {
            "choices": [
                {
                    "text": "a",
                    "finish_reason": "stop",
                    "logprobs": {"tokens": ["token_id:11"], "token_logprobs": [-0.1]},
                }
            ],
            "nvext": {
                "completion_token_ids": [11],
                "completion_token_logprobs": [-0.1],
            },
        },
    )

    assert generation.token_ids == [11]
    assert generation.token_logprobs == [-0.1]


def test_responses_stream_uses_cumulative_nvext_and_s3_metadata():
    generation = DynamoGeneration(api_mode="responses")
    generation.consume_sse(
        "response.output_text.delta",
        {
            "delta": "a",
            "nvext": {
                "completion_token_ids": [11],
                "completion_token_logprobs": [-0.1],
            },
        },
    )
    generation.consume_sse(
        "response.output_text.delta",
        {
            "delta": "b",
            "nvext": {
                "completion_token_ids": [11, 12],
                "completion_token_logprobs": [-0.1, -0.2],
            },
        },
    )
    generation.consume_sse(
        "response.completed",
        {
            "response": {
                "id": "resp_next",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ab"}],
                    }
                ],
                "nvext": {
                    "completion_token_ids": [11, 12],
                    "completion_token_logprobs": [-0.1, -0.2],
                },
            }
        },
    )
    generation.apply_metadata(
        {
            "output_token_logprobs": [
                [-0.1, 11, "a"],
                [-0.2, 12, "b"],
            ],
            "finish_reason": {"type": "stop"},
        }
    )
    generation.align_logprobs(required=True)

    assert generation.token_ids == [11, 12]
    assert generation.token_logprobs == [-0.1, -0.2]
    assert generation.text == "ab"
    assert generation.response_id == "resp_next"
    assert generation.finish_reason == "stop"
    assert generation.terminal_event_received is True

    generation.apply_metadata(
        {
            "output_token_logprobs": [[-0.1, 11, "a"], [-0.2, 12, "b"]],
            "finish_reason": {"type": "abort"},
        }
    )
    assert generation.finish_reason == "stop"


def test_responses_created_plus_s3_finish_is_not_a_persisted_terminal_event():
    generation = DynamoGeneration(api_mode="responses")
    generation.consume_sse(
        "response.created",
        {"response": {"id": "resp_unpersisted", "status": "in_progress", "output": []}},
    )
    generation.apply_metadata(
        {
            "output_token_logprobs": [[-0.1, 11, "a"]],
            "finish_reason": {"type": "stop"},
        }
    )

    assert generation.finish_reason == "stop"
    assert generation.terminal_event_received is False


def test_responses_content_filter_is_not_mapped_to_length():
    generation = DynamoGeneration(api_mode="responses")
    generation.consume_sse(
        "response.incomplete",
        {
            "response": {
                "id": "resp_filtered",
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
            }
        },
    )

    assert generation.finish_reason == "content_filter"
    assert generation.stop_reason == "content_filter"
    assert generation.terminal_event_received is True


def test_s3_metadata_cannot_extend_a_streamed_terminal_sequence():
    generation = DynamoGeneration(api_mode="responses")
    generation.token_ids = [11]
    generation.token_logprobs = [-0.1]

    generation.apply_metadata(
        {
            "output_token_logprobs": [
                [-0.1, 11, "a"],
                [-0.2, 12, "b"],
            ],
            "finish_reason": {"type": "abort"},
        }
    )

    assert generation.token_ids == [11]
    assert generation.token_logprobs == [-0.1]
    assert generation.finish_reason == "abort"


def test_s3_metadata_populates_tokens_when_no_stream_tokens_exist():
    generation = DynamoGeneration(api_mode="responses")
    generation.apply_metadata(
        {
            "output_token_logprobs": [
                [-0.1, 11, "a"],
                [-0.2, 12, "b"],
            ],
            "finish_reason": {"type": "stop"},
        }
    )

    assert generation.token_ids == [11, 12]
    assert generation.token_logprobs == [-0.1, -0.2]


def test_uploaded_metadata_populates_routed_experts_and_weight_version():
    sample = Sample(tokens=[1, 2, 3, 4])
    args = SimpleNamespace(num_layers=2, moe_router_topk=2)
    routed_experts = np.arange(12, dtype=np.int32).reshape(3, 2, 2)

    apply_uploaded_metadata_to_sample(
        sample,
        args,
        {
            "routed_experts": {
                "type": "ndarray",
                "dtype": "int32",
                "shape": [3, 2, 2],
                "data": routed_experts.tobytes(),
            },
            "weight_version": "policy-17",
        },
    )

    np.testing.assert_array_equal(sample.rollout_routed_experts, routed_experts)
    assert sample.weight_versions == ["policy-17"]


def test_uploaded_weight_version_does_not_require_moe_shape_args():
    sample = Sample(tokens=[1, 2])

    apply_uploaded_metadata_to_sample(
        sample,
        SimpleNamespace(),
        {"weight_version": "policy-18"},
    )

    assert sample.weight_versions == ["policy-18"]


def test_uploaded_routed_experts_are_trimmed_with_the_sample():
    sample = Sample(tokens=[1, 2, 3])
    args = SimpleNamespace(num_layers=1, moe_router_topk=2)
    routed_experts = np.arange(8, dtype=np.int32).reshape(4, 1, 2)

    apply_uploaded_metadata_to_sample(
        sample,
        args,
        {"routed_experts": routed_experts},
    )

    np.testing.assert_array_equal(sample.rollout_routed_experts, routed_experts[:2])


def test_uploaded_metadata_sequence_preserves_versions_and_uses_latest_routes():
    sample = Sample(tokens=[1, 2, 3, 4])
    args = SimpleNamespace(num_layers=1, moe_router_topk=2)
    routed_experts = np.arange(6, dtype=np.int32).reshape(3, 1, 2)

    apply_uploaded_metadata_sequence_to_sample(
        sample,
        args,
        [
            {"weight_version": "policy-1"},
            {"weight_version": "policy-2", "routed_experts": routed_experts},
        ],
    )

    assert sample.weight_versions == ["policy-1", "policy-2"]
    np.testing.assert_array_equal(sample.rollout_routed_experts, routed_experts)


def test_aborted_response_state_is_not_reused():
    from slime.rollout.sglang_rollout import _record_dynamo_response_state

    sample = Sample(metadata={"dynamo_previous_response_id": "resp_stale"})
    generation = DynamoGeneration(api_mode="responses", response_id="resp_aborted")

    _record_dynamo_response_state(sample, generation, reusable=False)

    assert sample.metadata["dynamo_response_id"] == "resp_aborted"
    assert "dynamo_previous_response_id" not in sample.metadata

    _record_dynamo_response_state(sample, generation, reusable=True)
    assert sample.metadata["dynamo_previous_response_id"] == "resp_aborted"


def test_cancelled_metadata_upload_is_optional(monkeypatch):
    from slime.rollout import sglang_rollout

    async def fail_read(*_args, **_kwargs):
        raise FileNotFoundError("upload was not finalized")

    monkeypatch.setattr(sglang_rollout, "read_uploaded_metadata_async", fail_read)

    async def exercise():
        assert await sglang_rollout._read_dynamo_metadata("s3://bucket/request", "msgpack", required=False) is None
        with pytest.raises(FileNotFoundError):
            await sglang_rollout._read_dynamo_metadata("s3://bucket/request", "msgpack", required=True)

    asyncio.run(exercise())


def test_short_routed_expert_suffix_is_padded_only_when_loss_masked():
    sample = Sample(tokens=[1, 2, 3, 4, 5], loss_mask=[1, 0, 0])
    args = SimpleNamespace(num_layers=1, moe_router_topk=2, num_experts=4)
    routed_experts = np.array([[[1, 2]], [[2, 3]]], dtype=np.int32)

    apply_uploaded_metadata_to_sample(
        sample,
        args,
        {"routed_experts": routed_experts},
    )

    assert sample.rollout_routed_experts.shape == (4, 1, 2)
    np.testing.assert_array_equal(sample.rollout_routed_experts[:2], routed_experts)
    assert np.all(sample.rollout_routed_experts[2:] < args.num_experts)


def test_short_routed_expert_suffix_rejects_trainable_tokens():
    sample = Sample(tokens=[1, 2, 3, 4], loss_mask=[1, 1])
    args = SimpleNamespace(num_layers=1, moe_router_topk=2, num_experts=4)

    with pytest.raises(ValueError, match="only covers"):
        apply_uploaded_metadata_to_sample(
            sample,
            args,
            {"routed_experts": np.array([[[1, 2]]], dtype=np.int32)},
        )


def test_generate_dynamo_stream_cancellation_keeps_prefix_without_s3(monkeypatch):
    from slime.rollout import sglang_rollout

    class FakeTokenizer:
        def encode(self, _prompt, add_special_tokens=False):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    state = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=None,
        active_dynamo_tasks=set(),
        aborted=False,
    )
    captured = {}

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield "event: response.created"
            yield "data: " + json.dumps({"response": {"id": "resp_partial", "status": "in_progress", "output": []}})
            yield "event: response.output_text.delta"
            state.aborted = True
            yield "data: " + json.dumps(
                {
                    "delta": "x",
                    "nvext": {
                        "completion_token_ids": [11, 12],
                        "completion_token_logprobs": [-0.1],
                    },
                }
            )

    class FakeClient:
        def stream(self, method, url, json, headers):
            captured.update(method=method, url=url, payload=json, headers=headers)
            return FakeResponse()

    async def unexpected_metadata_read(*_args, **_kwargs):
        raise AssertionError("cancelled streams must not wait for S3 metadata")

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout.http_utils, "_http_client", FakeClient())
    monkeypatch.setattr(sglang_rollout, "_read_dynamo_metadata", unexpected_metadata_read)

    args = SimpleNamespace(
        dynamo_api_mode="responses",
        sglang_router_ip="dynamo",
        sglang_router_port=3000,
        hf_checkpoint="model",
        dynamo_metadata_upload_url="s3://bucket/run",
        dynamo_metadata_upload_format="msgpack",
        dynamo_request_retries=1,
        dynamo_responses_store=True,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
    )
    sample = Sample(
        group_index=0,
        index=0,
        prompt="hello",
        metadata={"dynamo_previous_response_id": "resp_stale"},
    )

    result = asyncio.run(
        sglang_rollout._generate_dynamo(
            args,
            sample,
            {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 5,
                "sampling_seed": 9,
                "skip_special_tokens": False,
                "no_stop_trim": True,
                "spaces_between_special_tokens": False,
            },
        )
    )

    assert result.status == Sample.Status.ABORTED
    assert result.tokens == [1, 2, 11]
    assert result.rollout_log_probs == [-0.1]
    assert result.response == "<11>"
    assert result.metadata["dynamo_response_id"] == "resp_partial"
    assert "dynamo_previous_response_id" not in result.metadata
    assert captured["url"].endswith("/v1/responses")
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["seed"] == 9
    assert not state.active_dynamo_tasks


def test_dynamo_worker_streams_and_requests_routes_by_default(monkeypatch):
    from slime.backends.dynamo_utils import dynamo_engine

    captured = {}

    class FakeProcess:
        pass

    def fake_popen(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.delenv("DYN_SGL_FORCE_NONSTREAM", raising=False)
    monkeypatch.setattr(dynamo_engine.subprocess, "Popen", fake_popen)
    engine = dynamo_engine.DynamoEngine.__new__(dynamo_engine.DynamoEngine)
    engine.args = SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        hf_checkpoint="model",
        use_rollout_routing_replay=True,
        rollout_stream_interval=50,
        sglang_dp_size=1,
        sglang_pp_size=1,
        fp16=False,
        offload_rollout=False,
        dynamo_router_kv_events=False,
    )
    engine.num_gpus_per_engine = None
    engine.tp_size = 2
    engine.server_port = 30001
    engine.server_host = "127.0.0.1"
    engine._host = "0.0.0.0"
    engine._base_gpu_id = 0
    engine._disaggregation_bootstrap_port = None
    engine._discovery_backend = "file"
    engine.worker_type = "regular"
    engine.sglang_overrides = {}
    engine.rank = 0
    engine._wait_healthy = lambda timeout: None
    engine.flush_cache = lambda: None

    engine._launch_worker()

    assert "--enable-return-routed-experts" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--stream-interval") + 1] == "50"
    assert "DYN_SGL_FORCE_NONSTREAM" not in captured["env"]


@pytest.mark.parametrize("api_mode", ["responses", "completions"])
def test_terminal_dynamo_event_wins_global_abort_race(monkeypatch, api_mode):
    from slime.rollout import sglang_rollout

    class FakeTokenizer:
        def encode(self, _prompt, add_special_tokens=False):
            return [1, 2]

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    state = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        processor=None,
        active_dynamo_tasks=set(),
        aborted=False,
    )
    terminal_delivered = asyncio.Event()

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            if api_mode == "responses":
                yield "event: response.completed"
                payload = {
                    "response": {
                        "id": "resp_complete",
                        "status": "completed",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
                        "nvext": {
                            "completion_token_ids": [11],
                            "completion_token_logprobs": [-0.1],
                        },
                    }
                }
            else:
                payload = {
                    "choices": [
                        {
                            "text": "x",
                            "finish_reason": "stop",
                            "logprobs": {"tokens": ["token_id:11"], "token_logprobs": [-0.1]},
                        }
                    ],
                    "nvext": {"completion_token_ids": [11]},
                }
            yield "data: " + json.dumps(payload)
            terminal_delivered.set()
            await asyncio.Event().wait()

    class FakeClient:
        def stream(self, method, url, json, headers):
            return FakeResponse()

    monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
    monkeypatch.setattr(sglang_rollout.http_utils, "_http_client", FakeClient())

    args = SimpleNamespace(
        dynamo_api_mode=api_mode,
        sglang_router_ip="dynamo",
        sglang_router_port=3000,
        hf_checkpoint="model",
        dynamo_metadata_upload_url=None,
        dynamo_metadata_upload_format="msgpack",
        dynamo_request_retries=1,
        dynamo_responses_store=True,
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
    )
    sample = Sample(group_index=0, index=0, prompt="hello")

    async def exercise():
        task = asyncio.create_task(
            sglang_rollout._generate_dynamo(
                args,
                sample,
                {
                    "max_new_tokens": 8,
                    "temperature": 0.0,
                    "top_p": 1.0,
                },
            )
        )
        await terminal_delivered.wait()
        state.aborted = True
        task.cancel()
        return await task

    result = asyncio.run(exercise())

    assert result.status == Sample.Status.COMPLETED
    assert result.tokens == [1, 2, 11]
    assert result.rollout_log_probs == [-0.1]
    if api_mode == "responses":
        assert result.metadata["dynamo_previous_response_id"] == "resp_complete"
    else:
        assert "dynamo_previous_response_id" not in result.metadata


def test_clear_task_cancellation_supports_python_310_task_api():
    from slime.rollout.sglang_rollout import _clear_task_cancellation

    _clear_task_cancellation(SimpleNamespace())


def test_generate_and_rm_does_not_mask_terminal_samples():
    from slime.rollout.sglang_rollout import generate_and_rm

    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=False,
    )
    sample = Sample(
        response="done",
        response_length=2,
        status=Sample.Status.TRUNCATED,
        reward=1,
        loss_mask=[1, 1],
    )

    result = asyncio.run(generate_and_rm(args, sample, {}))

    assert result.loss_mask == [1, 1]


def test_resumed_partial_masks_only_the_existing_prefix(monkeypatch):
    from slime.rollout import sglang_rollout

    args = SimpleNamespace(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        group_rm=False,
        custom_generate_function_path=None,
    )
    sample = Sample(
        tokens=[1, 2],
        response="old",
        response_length=2,
        rollout_log_probs=[-0.1, -0.2],
        status=Sample.Status.ABORTED,
        reward=1,
    )

    async def fake_generate(_args, current_sample, _sampling_params):
        assert current_sample.loss_mask == [0, 0]
        current_sample.tokens.append(3)
        current_sample.response += "new"
        current_sample.response_length += 1
        current_sample.rollout_log_probs.append(-0.3)
        current_sample.loss_mask.append(1)
        current_sample.status = Sample.Status.TRUNCATED
        return current_sample

    async def exercise():
        state = SimpleNamespace(
            semaphore=asyncio.Semaphore(1),
            aborted=False,
            dp_rank_context=lambda: nullcontext(0),
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)
        monkeypatch.setattr(sglang_rollout, "generate", fake_generate)
        return await sglang_rollout.generate_and_rm(args, sample, {})

    result = asyncio.run(exercise())

    assert result.loss_mask == [0, 0, 1]
    assert result.rollout_log_probs == [-0.1, -0.2, -0.3]


def test_abort_buffers_only_nonempty_fully_aborted_groups(monkeypatch):
    from slime.rollout import sglang_rollout

    partial = Sample(response="x", response_length=1, status=Sample.Status.ABORTED)
    terminal = Sample(response="done", response_length=2, status=Sample.Status.TRUNCATED)
    empty = Sample(response="", response_length=0, status=Sample.Status.ABORTED)

    async def exercise():
        async def return_group(group):
            return group

        state = SimpleNamespace(
            aborted=False,
            active_dynamo_tasks=set(),
            pendings={
                asyncio.create_task(return_group([partial])),
                asyncio.create_task(return_group([terminal])),
                asyncio.create_task(return_group([empty])),
            },
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)

        return await sglang_rollout.abort(
            SimpleNamespace(rollout_backend="dynamo", partial_rollout=True),
            rollout_id=7,
        )

    assert asyncio.run(exercise()) == [[partial]]
    assert partial.metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
