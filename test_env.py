import os
from glob import glob
from pathlib import Path

import torch

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.version)

def check_cuda(cuda_base_path):
    return all(
        x.exists()
        for x in [
            cuda_base_path / "bin" / "cudart64_12.dll",
            cuda_base_path / "bin" / "ptxas.exe",
            cuda_base_path / "include" / "cuda.h",
            cuda_base_path / "lib" / "x64" / "cuda.lib",
        ]
    )

def find_cuda():
    cuda_base_path = os.environ.get("CUDA_PATH")
    print(cuda_base_path)
    if cuda_base_path is not None:
        cuda_base_path = Path(cuda_base_path)
        if not check_cuda(cuda_base_path):
            cuda_base_path = None

    if cuda_base_path is None:
        paths = glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*")
        # Heuristic to find the highest version
        paths = sorted(paths)[::-1]
        for path in paths:
            cuda_base_path = Path(path)
            if check_cuda(cuda_base_path):
                break
            else:
                cuda_base_path = None

    if cuda_base_path is None:
        print("WARNING: Failed to find CUDA.")
        return None, [], []

    return (
        str(cuda_base_path / "bin"),
        [str(cuda_base_path / "include")],
        [str(cuda_base_path / "lib" / "x64")],
    )

find_cuda()