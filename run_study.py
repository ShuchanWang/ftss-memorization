"""Run migrated studies using python run_study.py <script> [args]."""

import os
import runpy
import sys
import unittest

from study_layout import ROOT, STUDY_DIR, setup, source_path


def main():
    setup()
    # Existing study manifests resolve public checkpoint and dataset paths here.
    os.chdir(ROOT)
    args = sys.argv[1:]
    if not args or args == ["--list"]:
        for path in sorted(STUDY_DIR.glob("*.py")):
            if not path.name.startswith("test_"):
                print(path.name)
        return
    if args[0] == "test":
        names = args[1:] or [p.stem for p in sorted(STUDY_DIR.glob("test_*.py"))]
        suite = unittest.defaultTestLoader.loadTestsFromNames(names)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    path = source_path(args[0])
    if path.parent != STUDY_DIR:
        raise SystemExit("Use the watermark repository launcher for watermark tasks.")
    sys.argv = [str(path), *args[1:]]
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
