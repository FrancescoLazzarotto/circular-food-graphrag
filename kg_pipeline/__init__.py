"""Knowledge-graph construction pipeline."""

import site
from pathlib import Path


def _refuse_masked_environment() -> None:
    """Refuse to import when a user-site torch masks the active environment.

    Importing any stage pulls in gliner or sentence-transformers. With a torch
    installed under ``~/.local`` on the path, that import fails deep inside
    ``transformers`` with ``operator torchvision::nms does not exist``, which
    names neither the cause nor the fix. This check fails early with both.

    Raises:
        RuntimeError: If the user site-packages directory contains ``torch``.
    """
    if not site.ENABLE_USER_SITE:
        return
    user_site = site.getusersitepackages()
    paths = [user_site] if isinstance(user_site, str) else list(user_site)
    masking = next((p for p in paths if (Path(p) / "torch").exists()), None)
    if masking:
        raise RuntimeError(
            f"{masking} carries its own torch and masks this environment, so "
            "gliner and sentence-transformers cannot import. Re-run the command "
            "with PYTHONNOUSERSITE=1."
        )


_refuse_masked_environment()
