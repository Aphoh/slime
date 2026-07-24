import json
import sys
from pathlib import Path

import pytest


NUM_GPUS = 0


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "swebench-pro"
sys.path.insert(0, str(EXAMPLE_DIR))

import completions_direct_model  # noqa: E402
from completions_direct_model import DirectCompletionsConfig, DirectCompletionsModel  # noqa: E402


def _native_model(tokenizer):
    model = DirectCompletionsModel.__new__(DirectCompletionsModel)
    model.config = DirectCompletionsConfig(base_url="http://dynamo", tokenizer_path="unused", top_k=20)
    model.tokenizer = tokenizer
    return model


def test_direct_native_generate_returns_exact_tokens_logprobs_and_metadata(monkeypatch):
    class Tokenizer:

        def decode(self, token_ids, skip_special_tokens=False):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            yield b"data: " + json.dumps(
                {
                    "text": "<11>",
                    "output_ids": [11],
                    "meta_info": {
                        "finish_reason": None,
                        "output_token_logprobs": [[-0.1, 11, None]],
                        "prompt_tokens": 3,
                    },
                }
            ).encode()
            yield b"data: " + json.dumps(
                {
                    "text": "<12>",
                    "output_ids": [12],
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": "</tool_call>"},
                        "output_token_logprobs": [[-0.2, 12, None]],
                        "completion_tokens": 2,
                        "weight_version": "policy-9",
                    },
                }
            ).encode()
            yield b"data: [DONE]"

    posted = []

    def urlopen(request, timeout):
        posted.append((request.full_url, json.loads(request.data), dict(request.header_items()), timeout))
        return Response()

    monkeypatch.setattr(completions_direct_model.urllib.request, "urlopen", urlopen)
    model = _native_model(Tokenizer())

    result = model.complete_prompt_ids([1, 2, 3], max_tokens=4, x_request_id="trajectory:0")

    assert posted[0][0] == "http://dynamo/generate"
    assert posted[0][1]["rid"] == "trajectory:0:try:0"
    assert posted[0][1]["input_ids"] == [1, 2, 3]
    assert posted[0][1]["sampling_params"]["top_k"] == 20
    assert posted[0][2]["X-request-id"] == "trajectory:0:try:0"
    assert result["content"] == "<11><12>"
    assert result["extra"]["generated_token_ids"] == [11, 12]
    assert result["extra"]["token_logprobs"] == [-0.1, -0.2]
    assert result["extra"]["finish_reason"] == "stop"
    assert result["extra"]["stop_reason"] == "</tool_call>"
    assert result["extra"]["dynamo_metadata"][-1]["weight_version"] == "policy-9"


def test_direct_native_generate_retries_before_stream_output(monkeypatch):
    class Tokenizer:

        def decode(self, token_ids, skip_special_tokens=False):
            return ""

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            yield b"data: " + json.dumps({"output_ids": [], "meta_info": {"finish_reason": {"type": "stop"}}}).encode()
            yield b"data: [DONE]"

    request_ids = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        request_ids.append((payload["rid"], request.get_header("X-request-id")))
        if len(request_ids) == 1:
            raise RuntimeError("transient")
        return Response()

    monkeypatch.setattr(completions_direct_model.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(completions_direct_model.time, "sleep", lambda _seconds: None)
    model = _native_model(Tokenizer())
    model.config.retries = 2

    result = model.complete_prompt_ids([1, 2], max_tokens=1, x_request_id="trajectory:1")

    assert request_ids == [
        ("trajectory:1:try:0", "trajectory:1:try:0"),
        ("trajectory:1:try:1", "trajectory:1:try:1"),
    ]
    assert result["extra"]["finish_reason"] == "stop"


def test_direct_native_generate_rejects_output_ids_without_logprobs(monkeypatch):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            yield b"data: " + json.dumps({"output_ids": [11], "meta_info": {"finish_reason": None}}).encode()

    monkeypatch.setattr(completions_direct_model.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    model = _native_model(object())

    with pytest.raises(RuntimeError, match="without selected-token logprobs"):
        model.complete_prompt_ids([1, 2], max_tokens=2)
