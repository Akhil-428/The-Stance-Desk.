"""1-2: Python environment and package imports; 8: CPU/GPU availability."""
import importlib
import sys

import pytest


def test_python_version_supported():
    """transformers 5.x requires Python >= 3.11; the project is developed on 3.13."""
    assert sys.version_info[:2] >= (3, 11)


@pytest.mark.parametrize("mod", ["torch", "transformers", "safetensors", "tokenizers", "sklearn", "numpy",
                                 "pandas", "scipy", "streamlit", "plotly", "joblib"])
def test_imports(mod):
    importlib.import_module(mod)


def test_project_modules_import():
    for m in ["config", "preprocessing", "evaluation", "model", "features", "ui.pages", "ui.navigation"]:
        importlib.import_module(m)


def test_torch_cpu_works():
    import torch

    x = torch.ones(3, 3)
    assert float((x @ x).sum()) == 27.0


@pytest.mark.gpu
def test_torch_gpu_works():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    assert "sm_86" in torch.cuda.get_arch_list()
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    assert torch.isfinite(a @ a).all()
