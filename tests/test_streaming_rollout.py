import asyncio
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.rollout.streaming_utils import merge_stream_chunk
from slime.utils.types import Sample

NUM_GPUS = 0


def _chunk(text, pairs, output_length):
    return {
        "text": text,
        "meta_info": {
            "output_token_logprobs": pairs,
            "output_token_logprobs_length": output_length,
        },
    }


def test_merge_cumulative_stream_chunks():
    tokens, log_probs, text = merge_stream_chunk(
        tokens=[],
        log_probs=[],
        text="",
        chunk=_chunk("a", [[-0.1, 11, None]], 1),
    )
    tokens, log_probs, text = merge_stream_chunk(
        tokens=tokens,
        log_probs=log_probs,
        text=text,
        chunk=_chunk("ab", [[-0.1, 11, None], [-0.2, 12, None]], 2),
    )

    assert tokens == [11, 12]
    assert log_probs == [-0.1, -0.2]
    assert text == "ab"


def test_merge_disjoint_stream_chunks():
    tokens, log_probs, text = merge_stream_chunk(
        tokens=[],
        log_probs=[],
        text="",
        chunk=_chunk("a", [[-0.1, 11, None]], 1),
    )
    tokens, log_probs, text = merge_stream_chunk(
        tokens=tokens,
        log_probs=log_probs,
        text=text,
        chunk=_chunk("b", [[-0.2, 12, None]], 2),
    )
    tokens, log_probs, text = merge_stream_chunk(
        tokens=tokens,
        log_probs=log_probs,
        text=text,
        chunk=_chunk("", [], 2),
    )

    assert tokens == [11, 12]
    assert log_probs == [-0.1, -0.2]
    assert text == "ab"


def test_merge_stream_rejects_inconsistent_length():
    with pytest.raises(ValueError, match="output_token_logprobs_length"):
        merge_stream_chunk(
            tokens=[11],
            log_probs=[-0.1],
            text="a",
            chunk=_chunk("bc", [[-0.2, 12, None], [-0.3, 13, None]], 4),
        )


def test_merge_cumulative_stream_preserves_text_when_intermediate_text_is_null():
    tokens, log_probs, text = merge_stream_chunk(
        tokens=[11],
        log_probs=[-0.1],
        text="a",
        chunk=_chunk(None, [[-0.1, 11, None]], 1),
    )

    assert tokens == [11]
    assert log_probs == [-0.1]
    assert text == "a"


def test_stream_cancellation_closes_request_and_keeps_prefix(monkeypatch):
    try:
        import sglang_router  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "sglang_router", SimpleNamespace(__version__="0.3.0"))

    try:
        import transformers  # noqa: F401
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoProcessor=object,
                AutoTokenizer=object,
                PreTrainedTokenizerBase=object,
                ProcessorMixin=object,
            ),
        )

    from slime.rollout import sglang_streaming_rollout as streaming

    first_chunk_seen = asyncio.Event()
    request_closed = False
    state = SimpleNamespace(
        tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=False: "".join(f"<{token_id}>" for token_id in token_ids)
        ),
        processor=None,
        aborted=False,
        streaming_generation=False,
        streaming_tasks=set(),
    )

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            nonlocal request_closed
            request_closed = True

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield "data: " + json.dumps(
                {
                    "text": None,
                    "meta_info": {
                        "output_token_logprobs": [[-0.1, 11, None]],
                        "output_token_logprobs_length": 1,
                    },
                }
            )
            first_chunk_seen.set()
            await asyncio.Event().wait()

    class FakeClient:
        def stream(self, method, url, json, headers):
            assert method == "POST"
            assert url == "http://frontend:8000/generate"
            assert json["stream"] is True
            return FakeResponse()

    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)
    monkeypatch.setattr(streaming, "_prepare_prompt_ids", lambda *_args: [1, 2])
    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())
    monkeypatch.setattr(
        streaming,
        "trace_span",
        lambda *_args, **_kwargs: nullcontext(SimpleNamespace(update=lambda *_args, **_kwargs: None)),
    )

    args = SimpleNamespace(
        ci_test=False,
        sglang_router_ip="frontend",
        sglang_router_port=8000,
        use_rollout_routing_replay=False,
        router_policy=None,
    )
    sample = Sample(prompt="hello")

    async def exercise():
        task = asyncio.create_task(streaming.generate_streaming(args, sample, {"max_new_tokens": 8}))
        await first_chunk_seen.wait()
        state.aborted = True
        task.cancel()
        return await task

    result = asyncio.run(exercise())

    assert result.status == Sample.Status.ABORTED
    assert result.tokens == [1, 2, 11]
    assert result.rollout_log_probs == [-0.1]
    assert result.response == "<11>"
    assert request_closed is True
    assert state.streaming_generation is True
    assert not state.streaming_tasks


def test_partial_abort_buffers_only_nonempty_fully_aborted_groups(monkeypatch):
    from slime.rollout import sglang_rollout

    partial = Sample(response="x", response_length=1, status=Sample.Status.ABORTED)
    terminal = Sample(response="done", response_length=2, status=Sample.Status.TRUNCATED)
    empty = Sample(response="", response_length=0, status=Sample.Status.ABORTED)

    async def exercise():
        async def return_group(group):
            return group

        state = SimpleNamespace(
            aborted=False,
            streaming_generation=True,
            streaming_tasks=set(),
            pendings={
                asyncio.create_task(return_group([partial])),
                asyncio.create_task(return_group([terminal])),
                asyncio.create_task(return_group([empty])),
            },
        )
        monkeypatch.setattr(sglang_rollout, "GenerateState", lambda _args: state)

        return await sglang_rollout.abort(
            SimpleNamespace(partial_rollout=True),
            rollout_id=7,
        )

    assert asyncio.run(exercise()) == [[partial]]
    assert partial.metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
