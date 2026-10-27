"""Package committed public sources, newly built wheels and verified frozen models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TAG = "latest-model-v6-v2.0.0-beta.7"
MODEL_DIRECTORY = f"models/{TAG}"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(models: Path, output: Path):
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT
    )
    if status:
        raise ValueError("Commit tracked changes before building a traceable release")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = json.loads(
        (ROOT / "models/assistant-release-v2.json").read_text(encoding="utf-8")
    )
    if manifest["release"] != TAG:
        raise ValueError("Wrong runtime manifest version")
    for record in manifest["models"]:
        path = models / record["filename"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Frozen model mismatch: {path.name}")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "package"
    package.mkdir()
    # Read committed blobs, never collect untracked local/private assets.
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--name-only", commit], cwd=ROOT
    )
    for raw in tracked.split(b"\0"):
        if not raw:
            continue
        name = raw.decode("utf-8")
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        )
    wheels = package / "wheels"
    for project in [ROOT, ROOT / "apps/ephemeral-assistant"]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                str(wheels),
                str(project),
            ],
            check=True,
        )
    for record in manifest["models"]:
        destination = package / MODEL_DIRECTORY / record["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(models / record["filename"], destination)
        shutil.copyfile(destination, output / record["filename"])
    shutil.copyfile(package / "scripts/install_beta.ps1", package / "install.ps1")
    shutil.copyfile(
        package / "releases/start-assistant.cmd", package / "start-assistant.cmd"
    )
    files = [
        {
            "path": p.relative_to(package).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in sorted(package.rglob("*"))
        if p.is_file()
    ]
    payload = {
        "schema_version": 1,
        "release": TAG,
        "source_commit": commit,
        "research_version": "0.1.1",
        "assistant_version": "0.2.0b7",
        "default_input": "packet",
        "default_estimator": "exact_conditional_posterior",
        "default_prior": "world-model-v2-S.json",
        "actionable": False,
        "files": files,
    }
    (package / "beta-release-manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(
        package / "beta-release-manifest.json", output / "beta-release-manifest.json"
    )
    for wheel in wheels.glob("*.whl"):
        shutil.copyfile(wheel, output / wheel.name)
    for name, source in {
        "assistant-release-v2.json": "models/assistant-release-v2.json",
        "LATEST_MODEL_CARD.md": "models/LATEST_MODEL_CARD.md",
        "RELEASE_NOTES.md": "releases/beta.7.md",
    }.items():
        shutil.copyfile(package / source, output / name)
    archive = output / "auction-moment-beta.7-windows.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        for p in sorted(package.rglob("*")):
            if p.is_file():
                entry = zipfile.ZipInfo(
                    p.relative_to(package).as_posix(), date_time=(2026, 10, 27, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o644 << 16
                z.writestr(entry, p.read_bytes())
    print(
        json.dumps(
            {
                "release": TAG,
                "source_commit": commit,
                "packaged_files": len(files),
                "zip_sha256": digest(archive),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.models.resolve(), args.output.resolve())
