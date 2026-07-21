import importlib.util
import sys
from pathlib import Path


def torch_cuda_lib_dirs() -> list[str]:
    """Directories of torch's bundled NVIDIA libs, for a subprocess's ``LD_LIBRARY_PATH``.

    Transformer Engine ``dlopen``s ``libnccl.so.2`` / ``libcudnn`` at import. When torch is a pip
    wheel these live under ``site-packages/nvidia/*/lib`` and aren't on the loader path by default,
    so TE fails to load. We return those dirs (plus ``torch/lib``) to prepend. In a system-CUDA
    environment (e.g. the cluster's NGC container) they don't exist and we return ``[]``, letting
    the container's own loader path apply.
    """
    spec = importlib.util.find_spec("torch")
    if not spec or not spec.origin:
        return []
    site = Path(spec.origin).parent.parent  # .../site-packages/torch/__init__.py -> site-packages
    dirs = [str(p) for p in sorted((site / "nvidia").glob("*/lib")) if p.is_dir()]
    torch_lib = site / "torch" / "lib"
    if torch_lib.is_dir():
        dirs.append(str(torch_lib))
    return dirs


class MegatronLMNotVendoredError(RuntimeError):
    """Raised by ensure_on_path() when the Megatron-LM submodule hasn't been checked out."""


def _default_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Megatron-LM"


def megatron_root(root: Path | None = None) -> Path:
    """Return the validated vendored Megatron-LM/ directory.

    Args:
        root: Override for the Megatron-LM directory, mainly for testing in isolation from
            whatever this environment's real submodule state happens to be. Defaults to the
            actual vendored submodule path.

    Raises:
        MegatronLMNotVendoredError: if the directory doesn't exist (submodule not initialized).
    """
    root = root or _default_root()
    if not root.is_dir():
        raise MegatronLMNotVendoredError(
            f"{root} does not exist. Run 'git submodule update --init' first."
        )
    return root


def ensure_on_path(root: Path | None = None) -> None:
    """Adds the vendored Megatron-LM/ directory to sys.path (for in-process imports).

    Args:
        root: see :func:`megatron_root`.

    Raises:
        MegatronLMNotVendoredError: if the directory doesn't exist (submodule not
            initialized). Callers that want to skip rather than hard-fail (e.g. pytest
            modules that should skip cleanly when the submodule isn't there) should catch
            this explicitly.
    """
    path_str = str(megatron_root(root))
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
