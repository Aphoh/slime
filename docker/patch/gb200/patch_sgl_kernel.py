"""Patch sgl-kernel CMakeLists.txt to add a SGL_KERNEL_GB200_ONLY build option.

On arm64 GB200 nodes we only need datacenter Blackwell sm_100a kernels. The
upstream CUDA 13 path also emits sm_103a/sm_110a/sm_120a/sm_121a, which makes
the Docker layer much slower and increases peak compiler memory. Default OFF
preserves upstream behavior.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/src/sglang/sgl-kernel/CMakeLists.txt")
s = p.read_text()

marker = 'option(SGL_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)'
new_opt = (
    marker
    + "\n"
    + 'option(SGL_KERNEL_GB200_ONLY             "Build only for GB200/B200 (sm_100a). Skips sm_103a/sm_110a/sm_120a/sm_121a." OFF)'
)
assert marker in s, "marker 1 (SM100A option line) not found"
s = s.replace(marker, new_opt, 1)

old = 'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_SM100A)'
assert old in s, "marker 2 (CUDA 12.8 block) not found"
s = s.replace(old, "if (NOT SGL_KERNEL_GB200_ONLY)\n\n" + old, 1)

old2 = 'if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR SGL_KERNEL_ENABLE_FP4)'
assert old2 in s, "marker 3 (FP4 block) not found"
insert_before = """endif()  # NOT SGL_KERNEL_GB200_ONLY

if (SGL_KERNEL_GB200_ONLY)
    list(APPEND SGL_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100a,code=sm_100a"
        "--compress-mode=size"
    )
endif()

"""
s = s.replace(old2, insert_before + old2, 1)

p.write_text(s)
print(f"edits applied to {p}")
