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
        description="默认被动抓包、可选 OCR/YOLO、无会话持久化的实时研究助手（仅人工参考）"
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path(
            os.environ.get(
                "AUCTION_ASSISTANT_MODELS",
                "models/latest-model-v6-v2.0.0-beta.7",
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
        "--input",
        choices=("packet", "ocr"),
        default="packet",
        help="证据来源：packet 为被动抓包（默认，需 root 模拟器内的 tcpdump），ocr 为截图 OCR/YOLO",
    )
    parser.add_argument(
        "--packet-pcap",
        type=Path,
        help="抓包模式下改为回放 PCAP 文件，而不是实时抓包",
    )
    parser.add_argument("--packet-interface", default="wlan0", help="模拟器内抓包的网卡")
    parser.add_argument("--protocol-port", type=int, default=8002)
    parser.add_argument(
        "--event-codebook",
        type=Path,
        help="事件代码表；默认使用 --treasures 同目录的 event_catalog.csv",
    )
    parser.add_argument(
        "--v6-only",
        action="store_true",
        help="不使用默认的精确条件后验，只用冻结 v6 估值",
    )
    parser.add_argument(
        "--experimental-v2-fusion",
        action="store_true",
        help="改用实验性的 v2 原融合规则；不使用默认精确条件后验",
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
        "--experimental-exact-sampling",
        action="store_true",
        help="精确条件后验现为默认估值器，此选项保留以兼容旧命令",
    )
    parser.add_argument(
        "--exact-budget-seconds",
        type=float,
        default=8.0,
        help="精确条件采样每次估值的时间预算（秒）",
    )
    parser.add_argument(
        "--exact-lanes",
        type=int,
        default=4,
        help="精确条件采样使用的工作进程数",
    )
    parser.add_argument(
        "--exact-refine-seconds",
        type=float,
        default=45.0,
        help="精确条件采样在预算内未收敛时，后台继续细化的时间上限（秒）；0 为关闭",
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


def exact_enabled(args: argparse.Namespace) -> bool:
    """The exact posterior is the default unless v6-only or another v2 mode is chosen."""
    return not (args.v6_only or args.experimental_v2_fusion or args.experimental_conditional_sampling)


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
    from .latest_model import LatestV6V2Predictor
    from .observations import ObservationStore
    from .runtime import InferenceCoordinator
    from .ui import AssistantWindow

    if not 1 <= int(args.protocol_port) <= 65535:
        raise ValueError("--protocol-port 必须在 1 到 65535 之间")
    exact = exact_enabled(args)
    catalog = load_catalog(args.treasures)
    codebook = None
    if args.input == "packet":
        from .packet_session import load_codebook

        codebook = load_codebook(args.event_codebook or Path(args.treasures).parent / "event_catalog.csv")
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
        exact_sampling=exact,
        exact_budget_seconds=args.exact_budget_seconds,
        exact_lanes=args.exact_lanes,
        exact_refine_seconds=args.exact_refine_seconds,
    )
    store = ObservationStore()
    inference = InferenceCoordinator(predictor)
    try:
        if args.input == "packet":
            window = _packet_window(args, store, inference, catalog, codebook, exact)
        else:
            window = _ocr_window(args, store, inference, catalog, model_paths, exact)
        window.run()
    finally:
        inference.close()
    return 0


def _packet_window(args, store, inference, catalog, codebook, exact):
    from .packet_capture import AdbPacketCapture, PcapFilePacketSource
    from .packet_session import PacketGame, PacketMonitor
    from .ui import AssistantWindow

    def source(listener):
        if args.packet_pcap is not None:
            return PcapFilePacketSource(args.packet_pcap, listener, protocol_port=args.protocol_port)
        return AdbPacketCapture(
            listener, adb=args.adb, serial=args.serial,
            interface=args.packet_interface, protocol_port=args.protocol_port,
        )

    monitor = PacketMonitor(
        source, PacketGame(codebook, catalog), store,
        # The exact sampler starts on each round's public offer.
        on_preview=inference.prewarm_snapshot if exact else None,
    )
    return AssistantWindow(
        store=store, pipeline=None, inference=inference, catalog=catalog, monitor=monitor,
        title="Auction Moment · 抓包估值助手（仅人工参考）",
    )


def _ocr_window(args, store, inference, catalog, model_paths, exact):
    from .capture import AdbFrameSource
    from .ocr import FixedLayoutOCR
    from .runtime import CapturePipeline
    from .ui import AssistantWindow
    from .vision import UltralyticsVisionBackend, VisionRecognizer

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
    pipeline = CapturePipeline(
        source=source,
        store=store,
        ocr=ocr,
        vision=vision,
        # Exact mode starts sampling on the round's events while the map scrolls.
        on_round_events=(
            (lambda round_number, capture: inference.prewarm(store, round_number, capture))
            if exact
            else None
        ),
    )
    return AssistantWindow(
        store=store,
        pipeline=pipeline,
        inference=inference,
        catalog=catalog,
    )


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (ModelError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
