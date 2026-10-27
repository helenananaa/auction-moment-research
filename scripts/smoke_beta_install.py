"""Exercise installed wheels, frozen models and hidden UI without a device.

Run with the newly installed environment's Python. This intentionally does not
add the source tree to sys.path, connect ADB, or replay private observations.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.package.resolve()
    assert importlib.metadata.version("auction-moment-research") == "0.1.1"
    assert importlib.metadata.version("auction-moment-assistant") == "0.2.0b7"
    import auction_moment_assistant.cli as cli
    import auction_moment_research

    for module in (cli, auction_moment_research):
        assert (
            Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
        ), "Source tree shadowed installed wheel"
    defaults = cli.build_parser().parse_args([])
    assert defaults.input == "packet" and cli.exact_enabled(defaults)
    assert defaults.models.name == "latest-model-v6-v2.0.0-beta.7"
    from auction_moment_assistant.models import (
        load_release_manifest,
        verify_model_directory,
    )

    manifest = load_release_manifest()
    models = package / "models/latest-model-v6-v2.0.0-beta.7"
    paths = verify_model_directory(models, manifest)
    assert len(paths) == 5
    result = {
        "research_version": "0.1.1",
        "assistant_version": "0.2.0b7",
        "source_commit": json.loads(
            (package / "beta-release-manifest.json").read_text(encoding="utf-8")
        )["source_commit"],
        "installed_wheels_used": True,
        "model_hashes_verified": 5,
        "default_input": "packet",
        "default_estimator": "exact_conditional_posterior",
        "live_device_used": False,
        "private_data_used": False,
    }
    with tempfile.TemporaryDirectory(prefix="auction-beta-smoke-") as temp:
        os.environ["YOLO_CONFIG_DIR"] = str(Path(temp) / "yolo")
        os.environ["MPLCONFIGDIR"] = str(Path(temp) / "matplotlib")
        pcap = Path(temp) / "empty.pcap"
        pcap.write_bytes(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        import tkinter as tk
        from auction_moment_assistant.ui import AssistantWindow
        from auction_moment_assistant.exact_sampler import Evidence

        original_tk = tk.Tk

        def hidden_tk(*a, **kw):
            root = original_tk(*a, **kw)
            root.withdraw()
            return root

        def exercise(window):
            try:
                window.root.update_idletasks()
                window.root.update()
                assert len(window.catalog) == 120
                assert window.root.winfo_children()
                predictor = window.inference.predictor
                assert predictor.exact_sampling and predictor.exact_engine.lanes == 4
                estimate = predictor.exact_engine.sample(
                    Evidence(count={0: 1, 1: 1, 2: 3, 3: 3, 4: 2}),
                    20261027,
                    time_budget=8,
                    n_vectors=512,
                    worlds=4096,
                    min_ess=20,
                ).summary()
                assert estimate["status"] in ("ok", "low_ess")
                assert 0 < estimate["p10"] <= estimate["p50"] <= estimate["p90"]
                result["synthetic_posterior"] = {
                    k: estimate[k] for k in ("status", "ess", "p10", "p50", "p90")
                }
                result["hidden_window_constructed"] = True
                result["worker_processes_started"] = 4
            finally:
                window.close()

        parsed = cli.build_parser().parse_args(
            [
                "--models",
                str(models),
                "--treasures",
                str(package / "data/v1/core/treasures.csv"),
                "--packet-pcap",
                str(pcap),
            ]
        )
        with (
            patch.object(tk, "Tk", hidden_tk),
            patch.object(AssistantWindow, "run", exercise),
        ):
            assert cli.run(parsed) == 0
    result["success"] = True
    result["limits"] = (
        "Synthetic prediction and withdrawn-window event loop only; no visible layout, OCR accuracy, live capture or full-match validation."
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
