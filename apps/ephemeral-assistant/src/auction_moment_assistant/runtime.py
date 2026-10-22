from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time
import traceback
from typing import Callable

import cv2
import numpy as np

from .capture import CaptureError, FrameSource
from .observations import ObservationStore
from .ocr import FixedLayoutOCR, OcrCapture
from .predictor import PredictionResult
from .state_machine import (
    DebouncedState,
    RoundWatchStateMachine,
    ScreenState,
    classify_state,
    extract_features,
    has_final_result_signature,
)
from .vision import ROUND_GRID_CROP, VisionRecognizer


def is_final_result(frame_rgb: np.ndarray) -> bool:
    try:
        return has_final_result_signature(extract_features(frame_rgb))
    except ValueError:
        return False


def _map_fingerprint(frame_rgb: np.ndarray) -> np.ndarray:
    left, top, right, bottom = ROUND_GRID_CROP
    crop = np.asarray(frame_rgb, dtype=np.uint8)[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (196, 98), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )


def _fingerprint_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


@dataclass(frozen=True)
class CaptureOutcome:
    revision: int
    final_detected: bool
    ocr: OcrCapture | None
    map_item_count: int
    errors: tuple[str, ...]
    debug: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScanOutcome:
    revision: int
    round_number: int
    success: bool
    ocr: OcrCapture | None
    map_item_count: int
    viewport_count: int
    estimated_rows: int | None
    alignment_stable: bool
    errors: tuple[str, ...]
    debug: tuple[str, ...] = ()


class CapturePipeline:
    def __init__(
        self,
        *,
        source: FrameSource,
        store: ObservationStore,
        ocr: FixedLayoutOCR | None,
        vision: VisionRecognizer | None,
        scroll_wait: float = 0.20,
        max_scroll_pages: int = 12,
        same_threshold: float = 1.20,
        on_round_events: Callable[[int, OcrCapture], None] | None = None,
    ) -> None:
        self.source = source
        self.store = store
        self.ocr = ocr
        self.vision = vision
        # Called with the round's events as soon as the map top is captured,
        # before scrolling; the store is still updated only at the end.
        self.on_round_events = on_round_events
        self.scroll_wait = max(0.05, float(scroll_wait))
        self.max_scroll_pages = max(1, int(max_scroll_pages))
        self.same_threshold = max(0.0, float(same_threshold))

    def capture_once(
        self, *, expected_round: int, offset_rows: int
    ) -> CaptureOutcome:
        """Single-frame diagnostic path retained for tests and development."""

        frame = self.source.capture()
        self.store.set_frame(frame)
        if is_final_result(frame):
            revision = self.store.reset()
            return CaptureOutcome(revision, True, None, 0, ())
        errors: list[str] = []
        debug: list[str] = []
        ocr_capture = None
        if self.ocr is not None:
            try:
                ocr_capture = self.ocr.recognize(
                    frame, expected_round=int(expected_round)
                )
            except Exception as exc:
                errors.append(f"OCR: {exc}")
                debug.append("OCR capture failed\n" + traceback.format_exc())
        map_items = ()
        if self.vision is not None:
            try:
                map_items = self.vision.recognize(
                    frame, offset_rows=int(offset_rows)
                )
            except Exception as exc:
                errors.append(f"vision: {exc}")
                debug.append("Vision capture failed\n" + traceback.format_exc())
        revision = self.store.apply_capture(
            round_number=(
                ocr_capture.round_number
                if ocr_capture is not None
                else int(expected_round)
            ),
            events=ocr_capture.events if ocr_capture is not None else (),
            bankroll=ocr_capture.bankroll if ocr_capture is not None else None,
            map_items=map_items,
        )
        return CaptureOutcome(
            revision,
            False,
            ocr_capture,
            len(map_items),
            tuple(errors),
            tuple(debug),
        )

    def _recognize_events(
        self,
        frame: np.ndarray,
        round_number: int,
        errors: list[str],
        debug: list[str],
    ) -> OcrCapture | None:
        try:
            capture = self.ocr.recognize(frame, expected_round=round_number)
        except Exception as exc:
            errors.append(f"OCR: {exc}")
            debug.append("Round OCR failed\n" + traceback.format_exc())
            return None
        if not capture.round_verified:
            errors.append("轮次标题 OCR 未通过阈值，已按状态机轮次暂存")
        return capture

    def _round_frame(self) -> np.ndarray:
        frame = self.source.capture()
        observed = classify_state(extract_features(frame))
        if observed != ScreenState.ROUND_ACTIVE:
            raise CaptureError(f"地图扫描中止：画面已变为 {observed.value}")
        return frame

    def _swipe(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration_ms: int,
    ) -> None:
        swipe = getattr(self.source, "swipe_map", None)
        if not callable(swipe):
            raise CaptureError("当前画面源不支持受限地图滚动")
        swipe(start, end, duration_ms)
        time.sleep(self.scroll_wait)

    def scan_round(self, *, expected_round: int) -> ScanOutcome:
        """Scan a round entirely in RAM and restore the map to the top."""

        round_number = max(1, min(5, int(expected_round)))
        errors: list[str] = []
        debug: list[str] = []
        frames: list[np.ndarray] = []
        top_reached = False
        bottom_reached = False
        restored_top = False
        performed_swipe = False
        current: np.ndarray | None = None
        ocr_capture = None
        ocr_done = False
        try:
            current = self._round_frame()
            fingerprint = _map_fingerprint(current)
            for _ in range(self.max_scroll_pages):
                self._swipe((340, 220), (340, 630), 420)
                performed_swipe = True
                candidate = self._round_frame()
                candidate_fingerprint = _map_fingerprint(candidate)
                difference = _fingerprint_difference(
                    fingerprint, candidate_fingerprint
                )
                current = candidate
                fingerprint = candidate_fingerprint
                if difference < self.same_threshold:
                    top_reached = True
                    break
            if not top_reached:
                raise CaptureError("达到滚动上限但未确认地图顶部")

            frames.append(current.copy())
            if self.on_round_events is not None and self.ocr is not None:
                # Same frame and OCR as below, just earlier, so inference can
                # start while the map is scrolled.
                ocr_done = True
                ocr_capture = self._recognize_events(frames[0], round_number, errors, debug)
                if ocr_capture is not None:
                    try:
                        self.on_round_events(round_number, ocr_capture)
                    except Exception:
                        debug.append("Early inference start failed\n" + traceback.format_exc())
            for _ in range(self.max_scroll_pages):
                self._swipe((340, 560), (340, 320), 650)
                performed_swipe = True
                candidate = self._round_frame()
                candidate_fingerprint = _map_fingerprint(candidate)
                difference = _fingerprint_difference(
                    fingerprint, candidate_fingerprint
                )
                if difference < self.same_threshold:
                    bottom_reached = True
                    break
                frames.append(candidate.copy())
                current = candidate
                fingerprint = candidate_fingerprint
            if not bottom_reached:
                errors.append("达到滚动上限但未确认地图底部")
        except Exception as exc:
            errors.append(str(exc))
            debug.append("Round map scan failed\n" + traceback.format_exc())
        finally:
            if performed_swipe:
                try:
                    restore_frame = current
                    restore_fingerprint = (
                        _map_fingerprint(restore_frame)
                        if restore_frame is not None
                        else None
                    )
                    for _ in range(self.max_scroll_pages):
                        self._swipe((340, 220), (340, 630), 420)
                        candidate = self._round_frame()
                        candidate_fingerprint = _map_fingerprint(candidate)
                        difference = (
                            _fingerprint_difference(
                                restore_fingerprint, candidate_fingerprint
                            )
                            if restore_fingerprint is not None
                            else float("inf")
                        )
                        restore_fingerprint = candidate_fingerprint
                        if difference < self.same_threshold:
                            restored_top = True
                            break
                except Exception as exc:
                    errors.append(f"恢复地图顶部失败：{exc}")
                    debug.append("Map top restore failed\n" + traceback.format_exc())

        map_items = ()
        estimated_rows = None
        alignment_stable = False
        if frames:
            if self.ocr is not None and not ocr_done:
                ocr_capture = self._recognize_events(frames[0], round_number, errors, debug)
            if self.vision is not None:
                try:
                    recognition = self.vision.recognize_viewports(frames)
                    map_items = recognition.items
                    estimated_rows = recognition.estimated_rows
                    alignment_stable = recognition.alignment_stable
                    if not alignment_stable:
                        errors.append("地图视口对齐置信度不足，请在左侧复核")
                except Exception as exc:
                    errors.append(f"vision: {exc}")
                    debug.append("Viewport recognition failed\n" + traceback.format_exc())
            revision = self.store.apply_capture(
                round_number=round_number,
                events=ocr_capture.events if ocr_capture is not None else (),
                bankroll=ocr_capture.bankroll if ocr_capture is not None else None,
                map_items=map_items,
            )
        else:
            revision = self.store.revision

        complete = bool(
            frames and top_reached and bottom_reached and restored_top
        )
        if frames:
            revision = self.store.apply_automatic_scan_proof(
                round_number=round_number,
                map_rows=estimated_rows,
                complete=complete,
                pre_bid=complete,
            )
        success = bool(frames and top_reached and bottom_reached and restored_top)
        return ScanOutcome(
            revision=revision,
            round_number=round_number,
            success=success,
            ocr=ocr_capture,
            map_item_count=len(map_items),
            viewport_count=len(frames),
            estimated_rows=estimated_rows,
            alignment_stable=alignment_stable,
            errors=tuple(dict.fromkeys(errors)),
            debug=tuple(dict.fromkeys(debug)),
        )


@dataclass(frozen=True)
class MonitorUpdate:
    kind: str
    state: ScreenState
    round_number: int
    message: str
    scan: ScanOutcome | None = None
    details: tuple[str, ...] = ()


class EphemeralStateMonitor:
    """Background visual watcher with no file, packet, tap, or key APIs."""

    STATE_LABELS = {
        ScreenState.UNKNOWN: "正在识别窗口",
        ScreenState.LOBBY: "大厅待机",
        ScreenState.EVENT_SELECTION: "事件选择阶段",
        ScreenState.ROUND_ACTIVE: "轮次进行中",
        ScreenState.BID_ENTRY: "出价输入阶段",
        ScreenState.BID_CONFIRM: "出价确认阶段",
        ScreenState.ROUND_RANKING: "本轮排名",
        ScreenState.FINAL_RESULT: "终局",
    }

    def __init__(
        self,
        pipeline: CapturePipeline,
        store: ObservationStore,
        *,
        interval: float = 0.25,
        stable_frames: int = 2,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.interval = max(0.10, float(interval))
        self._debounced = DebouncedState(stable_frames)
        self._machine = RoundWatchStateMachine()
        self._updates: Queue[MonitorUpdate] = Queue()
        self._manual_scans: Queue[int] = Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="auction-window-watch", daemon=True
        )
        self._thread.start()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def request_scan(self, round_number: int) -> None:
        self._manual_scans.put(max(1, min(5, int(round_number))))

    def set_round(self, round_number: int) -> None:
        self._machine.set_round(round_number)
        self.store.set_round(round_number)

    def drain(self) -> tuple[MonitorUpdate, ...]:
        updates = []
        while True:
            try:
                updates.append(self._updates.get_nowait())
            except Empty:
                return tuple(updates)

    def _publish(
        self,
        kind: str,
        state: ScreenState,
        round_number: int,
        message: str,
        scan: ScanOutcome | None = None,
        details: tuple[str, ...] = (),
    ) -> None:
        self._updates.put(
            MonitorUpdate(
                kind,
                state,
                int(round_number),
                message,
                scan,
                tuple(details),
            )
        )

    def _run_scan(self, round_number: int, state: ScreenState) -> None:
        self._publish("scan_started", state, round_number, "正在扫描本轮地图…")
        outcome = self.pipeline.scan_round(expected_round=round_number)
        self._machine.mark_scan_result(round_number, outcome.success)
        message = (
            f"扫描完成：{outcome.viewport_count} 个视口，"
            f"{outcome.map_item_count} 个地图对象；进入预测待机"
            if outcome.success
            else "扫描未完整完成，可在轮次主界面手动重扫"
        )
        self._publish(
            "scan_finished",
            state,
            round_number,
            message,
            outcome,
            outcome.debug,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.10)
                continue
            try:
                manual_round = self._manual_scans.get_nowait()
            except Empty:
                manual_round = None
            if manual_round is not None:
                try:
                    self._run_scan(manual_round, ScreenState.ROUND_ACTIVE)
                except Exception as exc:
                    self._publish(
                        "error",
                        self._machine.state,
                        manual_round,
                        f"手动重扫失败：{exc}",
                        details=(traceback.format_exc(),),
                    )
                continue
            started = time.monotonic()
            try:
                frame = self.pipeline.source.capture()
                features = extract_features(frame)
                observed = classify_state(features)
                stable = self._debounced.update(observed)
                if stable != observed:
                    self._stop.wait(self.interval)
                    continue
                actions = self._machine.observe(stable, features)
                for action in actions:
                    if action.kind == "session_started":
                        self.store.reset()
                        self.store.set_round(action.round_number)
                    elif action.kind == "round_changed":
                        self.store.set_round(action.round_number)
                    elif action.kind == "session_ended":
                        self.store.reset()
                    if action.kind in {"state_changed", "round_changed"}:
                        self._publish(
                            action.kind,
                            action.state,
                            action.round_number,
                            self.STATE_LABELS[action.state],
                        )
                    elif action.kind == "session_started":
                        self._publish(
                            action.kind,
                            action.state,
                            action.round_number,
                            "检测到开局，开始自动监视第 1 轮",
                        )
                    elif action.kind == "session_ended":
                        self._publish(
                            action.kind,
                            action.state,
                            action.round_number,
                            (
                                "已回到大厅，本局临时数据已从内存清空"
                                if action.state == ScreenState.LOBBY
                                else "检测到终局，本局临时数据已从内存清空"
                            ),
                        )
                    elif action.kind == "scan_round":
                        self._run_scan(action.round_number, action.state)
            except Exception as exc:
                self._publish(
                    "error",
                    self._machine.state,
                    self._machine.round_number,
                    f"窗口监视失败：{exc}",
                    details=(traceback.format_exc(),),
                )
                self._stop.wait(0.75)
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval - elapsed))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class InferenceCoordinator:
    def __init__(self, predictor) -> None:
        self.predictor = predictor
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="auction-predictor"
        )

    def submit(self, store: ObservationStore) -> Future[PredictionResult]:
        snapshot = store.snapshot()
        return self._executor.submit(self.predictor.predict, snapshot)

    def prewarm(self, store: ObservationStore, round_number: int, capture: OcrCapture) -> bool:
        """Let a predictor that supports it start on the round's events early."""
        prewarm = getattr(self.predictor, "prewarm", None)
        if not callable(prewarm):
            return False
        return bool(prewarm(store.preview_snapshot(round_number, capture.events)))

    def refine(self, store: ObservationStore, result: PredictionResult) -> bool:
        """Keep refining a current estimate that ran out of time, if the predictor can."""
        refine = getattr(self.predictor, "refine", None)
        if (
            not callable(refine)
            or result.status != "exact_v2_posterior_low_ess"
            or not self.current(store, result)
        ):
            return False
        return bool(refine(int(result.revision)))

    def refined(self, store: ObservationStore) -> tuple[str, PredictionResult | None]:
        """Never blocks: ("running", None), ("done", result or None) or ("gone", None)."""
        poll = getattr(self.predictor, "poll_refined", None)
        if not callable(poll):
            return "gone", None
        state, result = poll()
        if result is not None and not self.current(store, result):
            return "gone", None
        return state, result

    @staticmethod
    def current(store: ObservationStore, result: PredictionResult) -> bool:
        return int(result.revision) == int(store.revision)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        close = getattr(self.predictor, "close", None)
        if callable(close):
            close()
