"""Patch SGLang 0.5.9 Qwen3.5 support for the SWE-Pro GB200 image.

This is intentionally narrow. Remove it once the base image carries an SGLang
build with the Qwen3.5 config init fix and non-eager vision FA3 import.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def _sglang_root() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("sglang is not importable")
    return Path(next(iter(spec.submodule_search_locations)))


def _replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1))
    return True


def patch_qwen35_config(root: Path) -> None:
    path = root / "srt" / "configs" / "qwen3_5.py"

    _replace_once(
        path,
        '''class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    model_type = "qwen3_5_moe"


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):''',
        '''class Qwen3_5MoeVisionConfig(Qwen3_5VisionConfig):
    model_type = "qwen3_5_moe"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):''',
    )
    _replace_once(
        path,
        '''class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"


class Qwen3_5MoeConfig(Qwen3_5Config):''',
        '''class Qwen3_5MoeTextConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_moe_text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Qwen3_5MoeConfig(Qwen3_5Config):''',
    )
    _replace_once(
        path,
        '''class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }
''',
        '''class Qwen3_5MoeConfig(Qwen3_5Config):
    model_type = "qwen3_5_moe"
    sub_configs = {
        "vision_config": Qwen3_5MoeVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
''',
    )


def patch_vision_fa3_import(root: Path) -> None:
    path = root / "srt" / "layers" / "attention" / "vision.py"
    old = "    from sgl_kernel.flash_attn import flash_attn_varlen_func"
    new = """    try:
        from sgl_kernel.flash_attn import flash_attn_varlen_func
    except Exception:
        flash_attn_varlen_func = None"""
    _replace_once(path, old, new)


def ensure_decord_stub(root: Path) -> None:
    try:
        import decord  # noqa: F401

        return
    except Exception:
        pass

    stub_path = root.parent / "decord.py"
    stub_path.write_text(
        '''class VideoReader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("decord is not installed in this text-only SWE-Pro image")


def cpu(*args, **kwargs):
    return None
'''
    )


def verify() -> None:
    from sglang.srt.configs.qwen3_5 import Qwen3_5MoeConfig

    cfg = Qwen3_5MoeConfig(text_config={"num_attention_heads": 32})
    assert hasattr(cfg.text_config, "num_attention_heads")

    if os.environ.get("SGLANG_QWEN35_IMPORT_VERIFY") == "1":
        import sglang.srt.layers.attention.vision  # noqa: F401
        import sglang.srt.models.qwen3_5  # noqa: F401
        import sglang.srt.multimodal.processors.qwen_vl  # noqa: F401


def main() -> None:
    root = _sglang_root()
    patch_qwen35_config(root)
    patch_vision_fa3_import(root)
    ensure_decord_stub(root)
    verify()
    print("patched SGLang Qwen3.5 compatibility")


if __name__ == "__main__":
    main()
