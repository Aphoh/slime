import sys
import types
from pathlib import Path


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
    assert payload["nvext"] == {"extra_fields": ["stop_reason"]}


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

    def _post(_url, json, timeout):
        posted_payloads.append(json)
        return _Response()

    monkeypatch.setattr(completions_direct_model.requests, "post", _post)
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused")
    model.tokenizer = object()

    result = model.complete_prompt_ids([1, 2], stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID])

    assert posted_payloads[0]["stop"] == ["</tool_call>"]
    assert "stop_token_ids" not in posted_payloads[0]
    assert posted_payloads[0]["nvext"] == {"extra_fields": ["stop_reason"]}
    assert result["extra"]["stop_reason"] == f"token_id:{GLM_TOOL_CLOSE_TOKEN_ID}"
    assert result["extra"]["generated_token_ids"] == [GLM_TOOL_CALL_TOKEN_ID]
