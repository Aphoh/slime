import copy
import json
import logging
import sys
import types
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLE_DIR))

import completions_direct_model  # noqa: E402
from completions_direct_model import (  # noqa: E402
    DirectCompletionsConfig,
    DirectCompletionsModel,
    GLM_EOS_TOKEN_ID,
    GLM_TOOL_CALL_TOKEN_ID,
    GLM_TOOL_CLOSE_TOKEN_ID,
    GLM_TOOL_STOPS,
    TOOL_CALL_END,
    encode_qwen_tool_observation_delta,
    glm_stop_strings_from_token_ids,
    parse_glm_tool_call_from_completion,
    stop_reason_token_ids,
)


def test_direct_completions_payload_maps_glm_stop_token_ids_to_stop_strings():
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")

    payload = model._build_payload_from_ids([1, 2, 3], max_tokens=4, stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID])

    assert payload["prompt"] == [1, 2, 3]
    assert payload["stop"] == ["</tool_call>"]
    assert "stop_token_ids" not in payload
    assert payload["nvext"] == {"extra_fields": ["stop_reason", "completion_token_ids"]}


def test_direct_completions_payload_can_ignore_eos_for_trace_replay():
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")

    payload = model._build_payload_from_ids([1, 2, 3], max_tokens=4, ignore_eos=True, min_tokens=4)

    assert payload["ignore_eos"] is True
    assert payload["min_tokens"] == 4


def test_direct_completions_payload_merges_dynamo_agent_context():
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    agent_context = {
        "session_type_id": "slime_swebench_pro",
        "session_id": "run-1",
        "trajectory_id": "run-1:swebench_pro:task:sample:0:id:sample-a",
    }

    payload = model._build_payload_from_ids([1, 2, 3], agent_context=agent_context)

    assert payload["nvext"]["extra_fields"] == ["stop_reason", "completion_token_ids"]
    assert payload["nvext"]["agent_context"] == agent_context
    assert "phase" not in payload["nvext"]["agent_context"]


def test_direct_completions_injects_retry_specific_metadata_upload_urls(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "<11>",
                        "finish_reason": "stop",
                        "logprobs": {"tokens": ["token_id:11"], "token_logprobs": [-0.1]},
                    }
                ],
                "nvext": {"completion_token_ids": [11]},
            }

    posted_payloads = []

    def _post(_url, json, headers, timeout):
        posted_payloads.append(copy.deepcopy(json))
        if len(posted_payloads) == 1:
            raise RuntimeError("transient")
        return _Response()

    monkeypatch.setattr(completions_direct_model.requests, "post", _post)
    monkeypatch.setattr(completions_direct_model.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        completions_direct_model,
        "_read_uploaded_metadata",
        lambda _url, _format: {
            "output_token_logprobs": [[-0.1, 11]],
            "finish_reason": {"type": "stop"},
        },
    )
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(
        base_url="http://dynamo",
        tokenizer_path="unused",
        retries=2,
        metadata_upload_url="s3://rollout-metadata/run-1",
    )
    model.tokenizer = _Tokenizer()

    result = model.complete_prompt_ids([1, 2], max_tokens=1, x_request_id="session:sample:turn:0")

    first = posted_payloads[0]["nvext"]["metadata_upload"]
    second = posted_payloads[1]["nvext"]["metadata_upload"]
    assert first["format"] == "msgpack"
    assert second["format"] == "msgpack"
    assert first["url"].startswith("s3://rollout-metadata/run-1/session-sample-turn-0-try-0-")
    assert second["url"].startswith("s3://rollout-metadata/run-1/session-sample-turn-0-try-1-")
    assert first["url"] != second["url"]
    assert result["extra"]["metadata_upload_url"] == second["url"]


def test_glm_stop_strings_from_token_ids_normalizes_known_stops():
    assert glm_stop_strings_from_token_ids([GLM_TOOL_CLOSE_TOKEN_ID, GLM_EOS_TOKEN_ID]) == GLM_TOOL_STOPS


def test_stop_reason_token_ids_normalizes_dynamo_values():
    assert stop_reason_token_ids(GLM_TOOL_CLOSE_TOKEN_ID) == [GLM_TOOL_CLOSE_TOKEN_ID]
    assert stop_reason_token_ids(f" token_id:{GLM_EOS_TOKEN_ID}\n") == [GLM_EOS_TOKEN_ID]
    assert stop_reason_token_ids("</tool_call>") == [GLM_TOOL_CLOSE_TOKEN_ID]
    assert stop_reason_token_ids("<|endoftext|>") == [GLM_EOS_TOKEN_ID]
    assert stop_reason_token_ids([GLM_TOOL_CLOSE_TOKEN_ID, f"token_id:{GLM_EOS_TOKEN_ID}", True]) == [
        GLM_TOOL_CLOSE_TOKEN_ID,
        GLM_EOS_TOKEN_ID,
    ]


def test_stop_reason_token_ids_uses_qwen_tokenizer_ids():
    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {
                "input_ids": {
                    "<tool_call>": [248058],
                    "</tool_call>": [248059],
                    "<|endoftext|>": [248044],
                }[text]
            }

    tokenizer = _Tokenizer()

    assert stop_reason_token_ids("</tool_call>", tokenizer) == [248059]
    assert stop_reason_token_ids("<|endoftext|>", tokenizer) == [248044]
    assert glm_stop_strings_from_token_ids([248059, 248044], tokenizer) == GLM_TOOL_STOPS


def test_tool_call_parse_is_gated_by_matched_stop_token():
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            pieces = {
                1: "thought",
                GLM_TOOL_CALL_TOKEN_ID: "<tool_call>",
                2: '{"name": "bash", "arguments": {"cmd": "ls"}}',
            }
            return "".join(pieces[token_id] for token_id in token_ids)

    model = types.SimpleNamespace(tokenizer=_Tokenizer())
    generated_ids = [1, GLM_TOOL_CALL_TOKEN_ID, 2]
    content = 'thought<tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}'

    _normal_text, tool_calls, needs_tool_close = parse_glm_tool_call_from_completion(
        model.tokenizer,
        content,
        generated_ids,
        [GLM_TOOL_CLOSE_TOKEN_ID],
    )
    assert needs_tool_close is True
    assert tool_calls[0]["function"]["name"] == "bash"

    normal_text, tool_calls, needs_tool_close = parse_glm_tool_call_from_completion(
        model.tokenizer,
        content,
        generated_ids,
        [GLM_EOS_TOKEN_ID],
    )
    assert normal_text == content
    assert tool_calls == []
    assert needs_tool_close is False


def test_tool_call_parse_uses_tokenizer_specific_qwen_tool_tokens():
    class _Tokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {
                "input_ids": {
                    "<tool_call>": [248058],
                    "</tool_call>": [248059],
                    "<|endoftext|>": [248044],
                }[text]
            }

        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            pieces = {
                1: "thought",
                248058: "<tool_call>",
                2: "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n",
                248059: "</tool_call>",
            }
            return "".join(pieces[token_id] for token_id in token_ids)

    tokenizer = _Tokenizer()
    generated_ids = [1, 248058, 2, 248059]
    content = "thought<tool_call><function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n"

    normal_text, tool_calls, needs_tool_close = parse_glm_tool_call_from_completion(
        tokenizer,
        content,
        generated_ids,
        stop_reason_token_ids(TOOL_CALL_END, tokenizer),
    )

    assert normal_text == "thought"
    assert needs_tool_close is False
    assert tool_calls == [
        {
            "id": "call_0_bash",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }
    ]


def test_qwen_tool_observation_delta_matches_chat_template_continuation():
    class _Tokenizer:
        captured = None

        def __call__(self, text, add_special_tokens=False):
            self.captured = text
            return {"input_ids": [1, 2, 3]}

    tokenizer = _Tokenizer()

    assert encode_qwen_tool_observation_delta(tokenizer, "README.md") == [1, 2, 3]
    assert tokenizer.captured == (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\nREADME.md\n</tool_response>"
        "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )


def test_direct_completions_carries_stop_reason(monkeypatch):
    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "<tool_call>",
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": [f"token_id:{GLM_TOOL_CALL_TOKEN_ID}"],
                            "token_logprobs": [-0.25],
                        },
                    }
                ],
                "nvext": {"stop_reason": f"token_id:{GLM_TOOL_CLOSE_TOKEN_ID}"},
            }

    posted_payloads = []
    posted_headers = []

    def _post(_url, json, headers, timeout):
        posted_payloads.append(json)
        posted_headers.append(headers)
        return _Response()

    monkeypatch.setattr(completions_direct_model.requests, "post", _post)
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = object()

    result = model.complete_prompt_ids([1, 2], stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID], x_request_id="traj:llm:0")

    assert posted_payloads[0]["stop"] == ["</tool_call>"]
    assert "stop_token_ids" not in posted_payloads[0]
    assert posted_payloads[0]["nvext"] == {"extra_fields": ["stop_reason", "completion_token_ids"]}
    assert posted_headers[0] == {"x-request-id": "traj:llm:0:try:0"}
    assert result["extra"]["stop_reason"] == f"token_id:{GLM_TOOL_CLOSE_TOKEN_ID}"
    assert result["extra"]["generated_token_ids"] == [GLM_TOOL_CALL_TOKEN_ID]


def test_direct_completions_clamps_oversized_backend_response(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "uncapped backend text",
                        "finish_reason": "stop",
                        "logprobs": {
                            "tokens": ["token_id:11", "token_id:12", "token_id:13", "token_id:14", "token_id:15"],
                            "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
                "nvext": {"stop_reason": "</tool_call>"},
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = _Tokenizer()

    result = model.complete_prompt_ids([1, 2], max_tokens=3)

    assert result["content"] == "<11><12><13>"
    assert result["extra"]["generated_token_ids"] == [11, 12, 13]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2, -0.3]
    assert result["extra"]["finish_reason"] == "length"
    assert result["extra"]["stop_reason"] is None
    assert result["extra"]["requested_max_tokens"] == 3
    assert result["extra"]["backend_generated_tokens"] == 5
    assert result["extra"]["locally_truncated_to_max_tokens"] is True
    assert result["extra"]["response"]["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}


def test_direct_completions_prefers_token_logprob_cardinality_when_tokens_overcount(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "inflated detokenized text",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": [
                                "token_id:11",
                                "token_id:12",
                                "token_id:13",
                                "token_id:14",
                                "token_id:15",
                            ],
                            "token_logprobs": [-0.1, -0.2, -0.3],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 99, "total_tokens": 101},
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = _Tokenizer()

    result = model.complete_prompt_ids([1, 2], max_tokens=3)

    assert result["content"] == "<11><12><13>"
    assert result["extra"]["generated_token_ids"] == [11, 12, 13]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2, -0.3]
    assert result["extra"]["backend_generated_tokens"] == 3
    assert result["extra"]["usage_completion_tokens"] == 99
    assert result["extra"]["parsed_generated_token_ids"] == 5
    assert result["extra"]["raw_token_logprob_count"] == 3
    assert result["extra"]["tokens_array_overcount"] is True
    assert result["extra"]["locally_truncated_to_max_tokens"] is False


def test_direct_completions_prefers_completion_token_ids_over_short_logprobs(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "full generated text",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": ["token_id:11", "token_id:12", "token_id:13"],
                            "token_logprobs": [-0.1, -0.2, -0.3],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
                "nvext": {"completion_token_ids": [11, 12, 13, 14, 15]},
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = _Tokenizer()

    result = model.complete_prompt_ids([1, 2], max_tokens=5)

    assert result["extra"]["generated_token_ids"] == [11, 12, 13, 14, 15]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2, -0.3, 0.0, 0.0]
    assert result["extra"]["backend_generated_tokens"] == 5
    assert result["extra"]["generated_token_source"] == "completion_token_ids"
    assert result["extra"]["raw_token_logprob_count"] == 3
    assert result["extra"]["tokens_array_overcount"] is False


def test_direct_completions_retry_uses_distinct_x_request_ids(monkeypatch):
    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "",
                        "finish_reason": "stop",
                        "logprobs": {"tokens": [], "token_logprobs": []},
                    }
                ],
            }

    posted_headers = []

    def _post(_url, json, headers, timeout):
        posted_headers.append(headers)
        if len(posted_headers) == 1:
            raise RuntimeError("transient")
        return _Response()

    monkeypatch.setattr(completions_direct_model.requests, "post", _post)
    monkeypatch.setattr(completions_direct_model.time, "sleep", lambda _seconds: None)
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused", retries=2)
    model.tokenizer = object()

    model.complete_prompt_ids([1, 2], x_request_id="traj:llm:7")

    assert posted_headers == [
        {"x-request-id": "traj:llm:7:try:0"},
        {"x-request-id": "traj:llm:7:try:1"},
    ]


def test_direct_completions_debug_logs_request_and_response(monkeypatch, caplog):
    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "choices": [
                    {
                        "text": "<11><12>",
                        "finish_reason": "length",
                        "logprobs": {
                            "tokens": ["token_id:11", "token_id:12"],
                            "token_logprobs": [-0.1, -0.2],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }

    monkeypatch.setenv("SWEPRO_COMPLETIONS_DEBUG", "1")
    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = object()

    caplog.set_level(logging.INFO, logger="completions_direct_model")
    result = model.complete_prompt_ids(
        [1, 2, 3],
        max_tokens=4,
        min_tokens=4,
        ignore_eos=True,
        agent_context={"trajectory_id": "traj-1", "session_id": "sess-1"},
        x_request_id="traj-1:llm:0",
    )

    debug_records = [
        json.loads(record.message.split(" ", 1)[1])
        for record in caplog.records
        if record.message.startswith(completions_direct_model.COMPLETIONS_DEBUG_LOG_PREFIX)
    ]

    assert [record["event"] for record in debug_records] == ["completion_request", "completion_response"]
    assert debug_records[0]["prompt_tokens"] == 3
    assert debug_records[0]["max_tokens"] == 4
    assert debug_records[0]["min_tokens"] == 4
    assert debug_records[0]["ignore_eos"] is True
    assert debug_records[0]["agent_trajectory_id"] == "traj-1"
    assert debug_records[1]["status_code"] == 200
    assert debug_records[1]["requested_max_tokens"] == 4
    assert debug_records[1]["usage_completion_tokens"] == 2
    assert debug_records[1]["backend_generated_tokens"] == 2
    assert debug_records[1]["generated_tokens"] == 2
    assert result["extra"]["generated_token_ids"] == [11, 12]
