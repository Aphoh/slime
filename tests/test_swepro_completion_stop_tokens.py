import sys
import types
from pathlib import Path



EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLE_DIR))

from completions_direct_model import (  # noqa: E402
    DirectCompletionsConfig,
    DirectCompletionsModel,
    GLM_EOS_TOKEN_ID,
    GLM_TOOL_CALL_TOKEN_ID,
    GLM_TOOL_CLOSE_TOKEN_ID,
    TOOL_CALL_END,
    encode_qwen_tool_observation_delta,
    parse_glm_tool_call_from_completion,
    stop_reason_token_ids,
)


def test_direct_completions_payload_uses_native_sglang_generate_fields():
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused", top_k=20, return_routed_experts=True)

    payload = model._build_payload_from_ids(
        [1, 2, 3],
        request_id="trajectory:0",
        max_tokens=4,
        ignore_eos=True,
        min_tokens=4,
        seed=123,
        stop=["</tool_call>"],
        stop_token_ids=[GLM_TOOL_CLOSE_TOKEN_ID],
    )

    assert payload["rid"] == "trajectory:0"
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["stream"] is True
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == -1
    assert payload["return_routed_experts"] is True
    assert payload["sampling_params"] == {
        "max_new_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "stop": ["</tool_call>"],
        "stop_token_ids": [GLM_TOOL_CLOSE_TOKEN_ID],
        "min_new_tokens": 4,
        "ignore_eos": True,
        "sampling_seed": 123,
    }


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
