"""Model inputs from captured visible events, and the monitor that drives the window.

For each round whose public offer and own personal result have arrived,
``PacketGame`` builds the two inputs the models were benchmarked on:

* an ObservationSnapshot made the same way as the 2026-10-19 controlled
  transcription replay (event semantics from the public event codebook, map
  items from the items the events revealed, the map height from the match
  opening), which feeds v6 through the usual adapter;
* the per-round decision record (events with their codebook fields and
  revealed items), which the exact sampler reads directly.

Everything stays in memory and is cleared at the next match opening.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Sequence

from .models import CatalogItem
from .observations import ObservationSnapshot, ObservationStore
from .packet_protocol import MATCH_OPENED, PERSONAL_EVENT_RESOLVED, ROUND_EVENT_OFFERED, PacketEvent


QUALITY_NAMES = {1: "白", 2: "蓝", 3: "紫", 4: "金", 5: "彩"}
REVEALS = frozenset({
    "reveal_identity_random", "reveal_position_quality_random", "reveal_position_random",
    "reveal_outline_by_size", "reveal_largest_area_item_outline",
    "reveal_outline_by_quality", "reveal_position_by_quality",
})
BY_QUALITY = frozenset({"reveal_position_by_quality", "reveal_outline_by_quality"})
CODEBOOK_FIELDS = ("effect", "event_template", "group_dimension", "group_value", "value_scale")


class PacketGameError(ValueError):
    pass


def load_codebook(path: Path) -> dict[int, dict]:
    """Event code -> static definition, from the public event catalog or instances table."""
    definitions: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            definition = {key: row[key] for key in CODEBOOK_FIELDS}
            previous = definitions.setdefault(int(row["event_code"]), definition)
            if definition != previous:
                raise PacketGameError(f"事件代码 {row['event_code']} 有多个定义")
    if not definitions:
        raise PacketGameError("事件代码表为空")
    return definitions


def event_semantics(definition: dict, packet: dict) -> dict:
    effect = definition["effect"]
    semantics = {"parsed": True, "effect": effect}
    dimension, group = definition["group_dimension"], definition["group_value"]
    if dimension == "quality":
        semantics["quality"] = group
    elif dimension == "size":
        width, height = map(int, group.lower().split("x"))
        semantics.update(width=width, height=height)
    if effect in REVEALS:
        if dimension == "random":
            semantics["count"] = int(group)
        return semantics
    observed = packet.get("observed_value")
    if observed is None:
        raise PacketGameError("数值事件缺少显示值")
    value, scale = int(observed), int(definition["value_scale"] or 1)
    if effect in {"quality_item_count", "size_item_count", "total_item_count"}:
        semantics["observed_count"] = value
    elif effect in {"quality_total_area", "size_total_area", "total_occupied_cells"}:
        semantics["observed_cells"] = value
    elif effect in {"quality_average_area", "size_average_area", "average_item_area"}:
        semantics["observed_average_cells"] = value / scale
    elif effect == "highest_quality":
        semantics["observed_quality"] = QUALITY_NAMES[value]
    else:
        semantics["observed_value"] = value
    return semantics


class PacketGame:
    """The current match's visible events, in capture order."""

    def __init__(self, codebook: dict[int, dict], catalog: Sequence[CatalogItem]) -> None:
        self.codebook = codebook
        self.catalog = tuple(catalog)
        self.reset()

    def reset(self) -> None:
        self.map_rows: int | None = None
        self.rounds: dict[int, dict] = {}
        self.current: int | None = None
        self.issues: list[str] = []

    def add(self, event: PacketEvent) -> str | None:
        """Record one event; returns "match_opened", "round_offered", "round_ready" or None."""
        if not event.valid:
            self.issues.append(f"undecodable_{event.kind}:{event.error}")
            return None
        payload = event.payload
        if event.kind == MATCH_OPENED:
            self.reset()
            rows = payload.get("map_rows")
            self.map_rows = int(rows) if rows and int(rows) > 0 else None
            if self.map_rows is None:
                self.issues.append("match_opened_without_map_rows")
            return "match_opened"
        if event.kind == ROUND_EVENT_OFFERED:
            number = payload.get("round")
            if number not in range(1, 6) or number in self.rounds:
                self.issues.append(f"unexpected_round_offer:{number}")
                return None
            self.rounds[number] = {"public": dict(payload["public_event"])}
            self.current = number
            return "round_offered"
        if event.kind == PERSONAL_EVENT_RESOLVED and int(payload.get("status") or 0) == 0:
            pair = self.rounds.get(self.current) if self.current else None
            if pair is None or "personal" in pair:
                self.issues.append("personal_event_without_open_round")
                return None
            pair["personal"] = {
                "code": payload.get("event_code"),
                "observed_value": payload.get("observed_value"),
                "items": payload.get("items") or [],
            }
            return "round_ready"
        return None

    def ready(self, round_number: int, *, partial: bool = False) -> bool:
        """All rounds up to this one are complete (the last needs only its public offer if partial)."""
        if self.map_rows is None:
            return False
        for number in range(1, round_number + 1):
            pair = self.rounds.get(number) or {}
            if "public" not in pair or ("personal" not in pair and not (partial and number == round_number)):
                return False
        return True

    def _packets(self, round_number: int, partial: bool):
        if not self.ready(round_number, partial=partial):
            raise PacketGameError(f"抓包没有覆盖本局第 1–{round_number} 轮（需从开局开始抓包）")
        for number in range(1, round_number + 1):
            for kind in ("public", "personal"):
                packet = self.rounds[number].get(kind)
                if packet is None:
                    continue
                definition = self.codebook.get(int(packet["code"]))
                if definition is None:
                    raise PacketGameError(f"事件代码 {packet['code']} 不在事件目录中")
                yield number, kind, packet, definition

    def decision(self, round_number: int, *, partial: bool = False) -> dict:
        """Per-round decision record: each event's codebook fields and revealed items."""
        events = []
        for number, kind, packet, definition in self._packets(round_number, partial):
            items = [{
                "index": int(raw["index"]),
                "quality": QUALITY_NAMES.get(int(raw.get("quality_code") or 0)),
                "value": int(raw["value"]) if raw.get("value") else None,
                "cells": sorted([int(c["row"]) - 1, int(c["column"]) - 1] for c in raw.get("cells") or []),
            } for raw in packet.get("items") or []]
            events.append({
                "round": number, "scope": kind, "code": int(packet["code"]),
                "effect": definition["effect"], "group_dimension": definition["group_dimension"],
                "group_value": definition["group_value"], "value_scale": int(definition["value_scale"] or 1),
                "observed_value": int(packet.get("observed_value") or 0),
                "reported_item_count": len(items),
                "reported_cell_count": sum(len(item["cells"]) for item in items),
                "quality_codes": sorted({code for code, name in QUALITY_NAMES.items()
                                         if any(item["quality"] == name for item in items)}),
                "items": items,
            })
        return {"round": int(round_number), "map_rows": self.map_rows, "map_rows_source": "match_opened",
                "events": events}

    def snapshot(self, round_number: int, *, partial: bool = False) -> ObservationSnapshot:
        """The round's snapshot; ``partial`` stops after the round's public offer (for prewarm)."""
        events, state = [], {}
        for number, kind, packet, definition in self._packets(round_number, partial):
            semantics = event_semantics(definition, packet)
            events.append({
                "round_number": number, "kind": kind, "text": definition["event_template"],
                "confidence": 1.0, "semantics": semantics, "source": "packet_capture",
                "human_locked": False, "suggested_text": None,
            })
            for raw in packet.get("items") or []:
                cells = {(int(c["row"]) - 1, int(c["column"]) - 1) for c in raw.get("cells") or []}
                if not cells or any(row < 0 or not 0 <= column < 10 for row, column in cells):
                    raise PacketGameError("揭示的藏品没有有效格子")
                item = state.setdefault(int(raw["index"]), {
                    "cells": set(), "quality": None, "quality_round": None, "value": None,
                    "size": None, "first_seen_round": number,
                })
                item["cells"].update(cells)
                quality = QUALITY_NAMES.get(int(raw.get("quality_code") or 0))
                if quality is None and semantics["effect"] in BY_QUALITY:
                    quality = semantics.get("quality")
                if quality is not None:
                    if item["quality"] not in (None, quality):
                        raise PacketGameError("同一藏品的品质前后矛盾")
                    item["quality"] = quality
                    if item["quality_round"] is None:
                        item["quality_round"] = number
                if raw.get("value") and int(raw["value"]) > 0:
                    if item["value"] not in (None, int(raw["value"])):
                        raise PacketGameError("同一藏品的价值前后矛盾")
                    item["value"] = int(raw["value"])
                if semantics["effect"] == "reveal_outline_by_size":
                    item["size"] = (int(semantics["width"]), int(semantics["height"]))
        map_items = []
        for item in state.values():
            row = min(r for r, _ in item["cells"])
            column = min(c for _, c in item["cells"])
            identity = None
            if item["value"] is not None:
                matching = [entry for entry in self.catalog if entry.value == item["value"]
                            and (item["quality"] is None or entry.quality == item["quality"])]
                if len(matching) != 1:
                    raise PacketGameError("揭示的价值无法唯一对应图鉴藏品")
                identity = matching[0]
                if item["size"] not in (None, (identity.width, identity.height)):
                    raise PacketGameError("藏品身份与揭示的轮廓矛盾")
                item["size"] = (identity.width, identity.height)
                item["quality"] = identity.quality
                item["quality_round"] = item["quality_round"] or item["first_seen_round"]
            width, height = item["size"] if item["size"] else (None, None)
            if width is not None and not item["cells"] <= {
                (row + dy, column + dx) for dy in range(height) for dx in range(width)
            }:
                raise PacketGameError("揭示的格子与已知形状不符")
            map_items.append({
                "row": row, "column": column, "width": width, "height": height,
                "marker_width": None, "marker_height": None,
                "quality": item["quality"], "catalog_id": identity.catalog_id if identity else None,
                "confidence": 1.0, "spatial": "complete" if identity else "outline" if width else "top_left",
                "source": "packet_capture", "human_locked": False, "identity_candidates": (),
                "first_seen_round": item["first_seen_round"], "quality_first_round": item["quality_round"],
            })
        if len({(item["row"], item["column"]) for item in map_items}) != len(map_items):
            raise PacketGameError("两件揭示藏品的左上角重合")
        return ObservationSnapshot(
            revision=0, round_number=int(round_number), bankroll=None,
            events=tuple(events), map_items=tuple(sorted(map_items, key=lambda i: (i["row"], i["column"]))),
            completeness_confirmed=True, completeness_source="packet_capture",
            map_rows=self.map_rows, map_height_exact=True, map_rows_source="packet_match_opened",
            pre_bid_confirmed=not partial,
            pre_bid_source="preview_before_personal_event" if partial else "packet_personal_event_resolved",
            packet_decision=self.decision(round_number, partial=partial),
        )


class PacketMonitor:
    """Drives the window from captured events, with the screen monitor's interface."""

    def __init__(
        self,
        source_factory: Callable[[Callable[[PacketEvent], None]], object],
        game: PacketGame,
        store: ObservationStore,
        *,
        on_preview: Callable[[ObservationSnapshot], object] | None = None,
    ) -> None:
        self.source_factory = source_factory
        self.game = game
        self.store = store
        self.on_preview = on_preview
        self.source = None
        self._lock = threading.RLock()
        self._updates: Queue = Queue()
        self._paused = threading.Event()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def start(self) -> None:
        self.source = self.source_factory(self._on_event)
        self.source.start()
        self._publish("packet_capture_started", self.store.round_number, "抓包已启动，等待开局")

    def set_paused(self, paused: bool) -> None:
        (self._paused.set if paused else self._paused.clear)()

    def request_scan(self, round_number: int) -> None:
        """Re-apply the captured evidence of a round (undoing manual edits to it)."""
        with self._lock:
            self._apply(int(round_number))

    def set_round(self, round_number: int) -> None:
        self.store.set_round(round_number)

    def drain(self) -> tuple:
        updates = []
        while True:
            try:
                updates.append(self._updates.get_nowait())
            except Empty:
                return tuple(updates)

    def close(self) -> None:
        if self.source is not None:
            self.source.stop()

    def _publish(self, kind: str, round_number: int, message: str, details: tuple[str, ...] = ()) -> None:
        from .runtime import MonitorUpdate
        from .state_machine import ScreenState

        state = ScreenState.LOBBY if kind == "session_started" else ScreenState.ROUND_ACTIVE
        self._updates.put(MonitorUpdate(kind, state, int(round_number), message, None, tuple(details)))

    def _apply(self, round_number: int) -> None:
        try:
            snapshot = self.game.snapshot(round_number)
        except (PacketGameError, KeyError, TypeError, ValueError) as exc:
            self._publish("packet_evidence_blocked", round_number, f"抓包证据不可用：{exc}",
                          tuple(self.game.issues[-5:]))
            return
        self.store.apply_packet_snapshot(snapshot)
        self._publish("scan_finished", round_number, f"第 {round_number} 轮抓包证据已就绪",
                      (f"events={len(snapshot.events)} map_items={len(snapshot.map_items)}",))

    def _on_event(self, event: PacketEvent) -> None:
        with self._lock:
            outcome = self.game.add(event)
            if outcome is None or self.paused:
                return
            if outcome == "match_opened":
                self.store.reset()
                self._publish("session_started", 1, f"抓包检测到开局：地图 {self.game.map_rows} 行")
            elif outcome == "round_offered":
                number = self.game.current
                self.store.set_round(number)
                self._publish("round_changed", number, f"第 {number} 轮公共事件已到达")
                if self.on_preview is not None and self.game.ready(number, partial=True):
                    try:
                        self.on_preview(self.game.snapshot(number, partial=True))
                    except (PacketGameError, KeyError, TypeError, ValueError):
                        pass  # the full snapshot reports the problem when the round is ready
            elif outcome == "round_ready":
                self._apply(self.game.current)

