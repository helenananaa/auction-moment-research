from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .models import (
    ModelError,
    download_models,
    load_catalog,
    load_release_manifest,
    verify_model_directory,
)


def default_treasures_path() -> Path:
    candidate = Path.cwd() / "data" / "v1" / "core" / "treasures.csv"
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="无抓包、无会话持久化的 OCR/YOLO 实时研究助手"
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(
            os.environ.get(
                "AUCTION_ASSISTANT_MODELS",
                "models/latest-model-v6-v2.0.0-beta.6",
            )
        ),
    )
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--model-base-url")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--verify-models", action="store_true")
    parser.add_argument("--treasures", type=Path, default=default_treasures_path())
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--minimum-ocr-confidence", type=float, default=0.85)
    parser.add_argument(
        "--experimental-v2-fusion",
        action="store_true",
        help="实验性启用 v2 原融合规则；默认使用 v6，v2 仅供诊断",
    )
    parser.add_argument(
        "--experimental-conditional-sampling",
        action="store_true",
        help="启用条件采样研究模式及严格有效样本融合门，尚待独立验证",
    )
    parser.add_argument(
        "--conditional-budget-seconds",
        type=float,
        default=2.0,
        help="v2 条件采样预算（秒）；未收敛时回退 v6",
    )
    parser.add_argument(
        "--bid-safety-factor",
        type=float,
        default=0.90,
        help="仅展示用建议最高出价系数，默认取 P10 的 90%%",
    )
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-vision", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    manifest = load_release_manifest(args.model_manifest)
    if args.download_models:
        model_paths = download_models(
            args.models,
            base_url=args.model_base_url,
            manifest=manifest,
        )
    else:
        model_paths = verify_model_directory(args.models, manifest)
    if args.verify_models:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "release": manifest["release"],
                    "models": {
                        role: path.name for role, path in sorted(model_paths.items())
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0
    from .capture import AdbFrameSource
    from .latest_model import LatestV6V2Predictor
    from .observations import ObservationStore
    from .ocr import FixedLayoutOCR
    from .runtime import CapturePipeline, InferenceCoordinator
    from .ui import AssistantWindow
    from .vision import UltralyticsVisionBackend, VisionRecognizer

    catalog = load_catalog(args.treasures)
    predictor = LatestV6V2Predictor(
        v6_model_path=model_paths["latest_value_model_v6"],
        world_model_v2_path=model_paths["generative_world_model_v2"],
        treasures_csv=args.treasures,
        catalog=catalog,
        minimum_ocr_confidence=args.minimum_ocr_confidence,
        bid_safety_factor=args.bid_safety_factor,
        conditional_budget_seconds=args.conditional_budget_seconds,
        conditional_sampling=args.experimental_conditional_sampling,
        experimental_v2_fusion=args.experimental_v2_fusion,
    )
    source = AdbFrameSource(adb=args.adb, serial=args.serial)
    ocr = (
        None
        if args.no_ocr
        else FixedLayoutOCR(minimum_confidence=args.minimum_ocr_confidence)
    )
    vision = None
    if not args.no_vision:
        backend = UltralyticsVisionBackend(model_paths, device=args.device)
        vision = VisionRecognizer(backend, catalog)
    store = ObservationStore()
    pipeline = CapturePipeline(source=source, store=store, ocr=ocr, vision=vision)
    window = AssistantWindow(
        store=store,
        pipeline=pipeline,
        inference=InferenceCoordinator(predictor),
        catalog=catalog,
    )
    window.run()
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (ModelError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
