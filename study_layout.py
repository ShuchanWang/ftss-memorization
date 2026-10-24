"""Paths for the migrated pretrained-flow fine-tuning studies."""

import os
from pathlib import Path
import sys


STUDY_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("WATERMARK_REPO", STUDY_ROOT.parent / "flow-matching-dynamics-watermarking")).resolve()
STUDY_DIR = STUDY_ROOT / "studies/flow_finetuning"


def setup():
    if not (ROOT / "repo_layout.py").is_file():
        raise RuntimeError("Set WATERMARK_REPO to the organized watermark checkout containing the model helpers.")
    sys.path.insert(0, str(ROOT))
    import repo_layout
    repo_layout.setup()
    paths = [str(STUDY_ROOT), str(STUDY_DIR)]
    sys.path[:0] = paths
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths + [p for p in inherited if p]))


def source_path(name):
    local = STUDY_DIR / name
    if local.is_file():
        return local
    import repo_layout
    return repo_layout.source_path(name)
