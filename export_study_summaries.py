"""Export compact FTSS study evidence; never upload checkpoints or owner keys."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from study_layout import ROOT, STUDY_ROOT, setup


def main():
    setup()
    from tools.export_run_summaries import compact
    output = STUDY_ROOT / "results/flow_finetuning"
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    names = {"config.json", "results.json", "summary.json", "analysis.json", "report.md",
             "summary.csv", "aggregate.json", "aggregate.csv", "associations.json"}
    for directory in sorted((STUDY_ROOT / "table_runs").iterdir()):
        if not directory.is_dir() or "smoke" in directory.name or "pilot" in directory.name:
            continue
        for source in sorted(directory.iterdir()):
            if source.name not in names or not source.is_file() or source.stat().st_size > 5_000_000:
                continue
            raw = source.read_bytes()
            if source.suffix == ".json":
                value = compact(compact(json.loads(raw), ROOT), STUDY_ROOT)
                data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
            elif source.suffix == ".csv":
                stream = io.StringIO()
                csv.writer(stream, lineterminator="\n").writerows(csv.reader(io.StringIO(raw.decode())))
                data = stream.getvalue().encode()
            else:
                data = raw.replace(str(ROOT).encode(), b"<watermark-repo>").replace(str(STUDY_ROOT).encode(), b"<ftss-repo>")
            destination = output / directory.name / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            entries.append(dict(source=str(source.relative_to(STUDY_ROOT)),
                                snapshot=str(destination.relative_to(STUDY_ROOT)),
                                source_sha256=hashlib.sha256(raw).hexdigest(),
                                snapshot_sha256=hashlib.sha256(data).hexdigest()))
    (output / "manifest.json").write_text(json.dumps(dict(files=entries), indent=2) + "\n")
    print(f"Exported {len(entries)} compact study files.")


if __name__ == "__main__":
    main()
