"""Knowledge-graph construction pipeline."""

import site
from pathlib import Path


def _refuse_masked_environment() -> None:
    """Fail with the cause, not with the symptom, when ``~/.local`` masks torch.

    Importing any stage pulls in gliner or sentence-transformers, and with a
    user-site torch on the path that import dies three frames deep in
    ``transformers`` on ``operator torchvision::nms does not exist`` — a message
    that names neither the cause nor the cure. The cure is the variable every
    command in this repository already carries.
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

# kg_pipeline package marker
