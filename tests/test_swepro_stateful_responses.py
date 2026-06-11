import copy
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

import completions_direct_model  # noqa: E402
from completions_direct_model import (  # noqa: E402
    DirectCompletionsConfig,
    DirectCompletionsModel,
    GLM_TOOL_CLOSE_TOKEN_ID,
)


def _responses_model(tokenizer=object()):
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
        stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID],
    )

    assert payload["input"] == response_input
    assert payload["previous_response_id"] == "resp_previous"
    assert payload["max_output_tokens"] == 4
    assert payload["ignore_eos"] is True
    assert payload["min_tokens"] == 4
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
