import copy
import sys
from pathlib import Path

import pytest


NUM_GPUS = 0


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

import completions_direct_model  # noqa: E402
from completions_direct_model import (  # noqa: E402
    DirectCompletionsConfig,
    DirectCompletionsModel,
    GLM_TOOL_CLOSE_TOKEN_ID,
)


_DEFAULT_TOKENIZER = object()


def _responses_model(tokenizer=_DEFAULT_TOKENIZER):
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(
        base_url="http://dynamo",
        tokenizer_path="unused",
        api_mode="responses",
        top_k=20,
    )
    model.tokenizer = tokenizer
    return model


def test_direct_responses_payload_carries_exact_tokens_and_state():
    model = _responses_model()
    response_input = [
        {
            "type": "function_call_output",
            "call_id": "call_server_1",
            "output": "README.md",
        }
    ]

    payload = model._build_payload_from_ids(
        [1, 2, 3],
        response_input=response_input,
        previous_response_id="resp_previous",
        max_tokens=4,
        ignore_eos=True,
        min_tokens=4,
        seed=123,
        skip_special_tokens=False,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
        stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID],
    )

    assert payload["input"] == response_input
    assert payload["previous_response_id"] == "resp_previous"
    assert payload["max_output_tokens"] == 4
    assert payload["ignore_eos"] is True
    assert payload["min_tokens"] == 4
    assert payload["seed"] == 123
    assert payload["skip_special_tokens"] is False
    assert payload["no_stop_trim"] is True
    assert payload["spaces_between_special_tokens"] is False
    assert payload["top_k"] == 20
    assert payload["stop"] == ["</tool_call>"]
    assert payload["include"] == ["message.output_text.logprobs"]
    assert payload["nvext"]["token_data"] == [1, 2, 3]
    assert payload["nvext"]["extra_fields"] == [
        "stop_reason",
        "completion_token_ids",
        "completion_token_logprobs",
    ]


def test_direct_responses_returns_exact_tokens_logprobs_and_server_call_id(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "resp_next",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_server_1",
                        "name": "bash",
                        "arguments": '{"command":"ls"}',
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                "nvext": {
                    "completion_token_ids": [11, 12],
                    "completion_token_logprobs": [-0.1, -0.2],
                    "stop_reason": "</tool_call>",
                },
            }

    posted = []

    def _post(url, json, headers, timeout):
        posted.append((url, copy.deepcopy(json), headers))
        return _Response()

    monkeypatch.setattr(completions_direct_model.requests, "post", _post)
    model = _responses_model(_Tokenizer())

    result = model.complete_prompt_ids(
        [1, 2, 3],
        response_input=[{"role": "user", "content": "hello"}],
        previous_response_id="resp_previous",
        max_tokens=4,
    )

    assert posted[0][0] == "http://dynamo/v1/responses"
    assert posted[0][1]["nvext"]["token_data"] == [1, 2, 3]
    assert posted[0][1]["previous_response_id"] == "resp_previous"
    assert result["content"] == "<11><12>"
    assert result["extra"]["response_id"] == "resp_next"
    assert result["extra"]["generated_token_ids"] == [11, 12]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2]
    assert result["extra"]["finish_reason"] == "stop"
    assert result["extra"]["response_tool_calls"][0]["id"] == "call_server_1"


def test_direct_responses_uses_streamed_prefix_with_longer_uploaded_metadata(monkeypatch):
    class _Tokenizer:
        @staticmethod
        def decode(token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "resp_next",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
                "nvext": {
                    "completion_token_ids": [11, 12],
                    "completion_token_logprobs": [-9.0, -9.0],
                    "stop_reason": " ",
                },
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(
        completions_direct_model,
        "_read_uploaded_metadata",
        lambda *_args, **_kwargs: {
            "output_token_logprobs": [[-0.1, 11], [-0.2, 12], [-0.3, 13]],
            "finish_reason": {"type": "abort"},
        },
    )
    model = _responses_model(_Tokenizer())
    model.config.metadata_upload_url = "s3://rollout-metadata/test"

    result = model.complete_prompt_ids(
        [1, 2],
        response_input=[{"role": "user", "content": "hello"}],
        max_tokens=4,
    )

    assert result["content"] == "<11><12>"
    assert result["extra"]["generated_token_ids"] == [11, 12]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2]
    assert result["extra"]["generated_token_source"] == "completion_token_ids"
    assert result["extra"]["finish_reason"] == "stop"
    assert result["extra"]["stop_reason"] == " "


def test_direct_responses_preserves_content_filter_reason(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "resp_filtered",
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
                "output": [],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                "nvext": {
                    "completion_token_ids": [11],
                    "completion_token_logprobs": [-0.1],
                },
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = _responses_model()

    result = model.complete_prompt_ids(
        [1, 2],
        response_input=[{"role": "user", "content": "hello"}],
        max_tokens=4,
    )

    assert result["extra"]["finish_reason"] == "content_filter"
    assert result["extra"]["stop_reason"] == "content_filter"


def test_direct_responses_rejects_missing_token_logprobs(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "id": "resp_next",
                "status": "completed",
                "output": [],
                "nvext": {"completion_token_ids": [11]},
            }

    monkeypatch.setattr(completions_direct_model.requests, "post", lambda *_args, **_kwargs: _Response())
    model = _responses_model()

    with pytest.raises(RuntimeError, match="without token logprobs"):
        model.complete_prompt_ids(
            [1, 2],
            response_input=[{"role": "user", "content": "hello"}],
            max_tokens=2,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
