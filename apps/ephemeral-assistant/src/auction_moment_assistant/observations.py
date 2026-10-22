from __future__ import annotations

import copy
import threading
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class EventObservation:
    round_number: int
    kind: str
    text: str
    confidence: float
    semantics: dict
    source: str = "ocr"
    human_locked: bool = False
    suggested_text: str | None = None


@dataclass
class MapObservation:
    row: int
    column: int
    width: int | None = None
    height: int | None = None
    # Detector/manual-review box geometry for anchor-only clues. These fields
    # are presentation metadata and must never be translated into known size.
    marker_width: int | None = None
    marker_height: int | None = None
    quality: str | None = None
    catalog_id: str | None = None
    confidence: float = 0.0
    spatial: str = "top_left"
    source: str = "vision"
    human_locked: bool = False
    identity_candidates: tuple[tuple[str, float], ...] = ()
    first_seen_round: int | None = None
    # Round in which a quality was first seen on this anchor; kept when a
    # later observation lacks the quality, so reveals can be dated.
    quality_first_round: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return int(self.row), int(self.column)


@dataclass(frozen=True)
class ObservationSnapshot:
    revision: int
    round_number: int
    bankroll: int | None
    events: tuple[dict, ...]
    map_items: tuple[dict, ...]
    completeness_confirmed: bool
    completeness_source: str
    map_rows: int | None
    map_height_exact: bool
    map_rows_source: str
    pre_bid_confirmed: bool
    pre_bid_source: str
    # Per-round decision record from packet capture; present only while the
    # store holds exactly what capture gave for this round (no edits since).
    packet_decision: dict | None = None


def _quality_first_round(
    observed: MapObservation, previous: MapObservation | None, round_number: int
) -> int | None:
    rounds = []
    if previous is not None and previous.quality_first_round is not None:
        rounds.append(int(previous.quality_first_round))
    if observed.quality is not None:
        rounds.append(
            int(observed.quality_first_round)
            if observed.quality_first_round is not None
            else int(round_number)
        )
    return min(rounds) if rounds else None


class ObservationStore:
    """Revisioned, memory-only evidence store.

    Human corrections win over later OCR/vision observations. The store has no
    serialization or path API by design.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._revision = 0
        self._round_number = 1
        self._bankroll: int | None = None
        self._bankroll_human_locked = False
        self._events: dict[tuple[int, str], EventObservation] = {}
        self._map_items: dict[tuple[int, int], MapObservation] = {}
        self._suppressed_map_keys: set[tuple[int, int]] = set()
        self._complete_rounds: dict[int, str] = {}
        self._map_extent_by_round: dict[
            int, tuple[int | None, bool, str]
        ] = {}
        self._pre_bid_rounds: dict[int, str] = {}
        self._packet_decision: tuple[int, int, dict] | None = None
        self._frame: np.ndarray | None = None

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def round_number(self) -> int:
        with self._lock:
            return self._round_number

    def _changed(self) -> int:
        self._revision += 1
        return self._revision

    def set_round(self, round_number: int) -> int:
        value = max(1, min(5, int(round_number)))
        with self._lock:
            if value == self._round_number:
                return self._revision
            self._round_number = value
            return self._changed()

    def set_frame(self, frame: np.ndarray) -> int:
        owned = np.asarray(frame, dtype=np.uint8).copy()
        with self._lock:
            self._zero_frame()
            self._frame = owned
            return self._changed()

    def frame_copy(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def apply_capture(
        self,
        *,
        round_number: int,
        events: Iterable[EventObservation],
        bankroll: int | None,
        map_items: Iterable[MapObservation],
    ) -> int:
        with self._lock:
            self._round_number = max(1, min(5, int(round_number)))
            for event in events:
                key = (int(event.round_number), str(event.kind))
                previous = self._events.get(key)
                if previous is not None and previous.human_locked:
                    previous.suggested_text = str(event.text)
                    continue
                self._events[key] = copy.deepcopy(event)
            if bankroll is not None and not self._bankroll_human_locked:
                self._bankroll = int(bankroll)
            for item in map_items:
                observed = copy.deepcopy(item)
                if observed.first_seen_round is None:
                    observed.first_seen_round = self._round_number
                key = observed.key
                if key in self._suppressed_map_keys:
                    continue
                previous = self._map_items.get(key)
                if previous is not None and previous.human_locked:
                    continue
                if previous is not None and previous.first_seen_round is not None:
                    observed.first_seen_round = min(
                        int(previous.first_seen_round),
                        int(observed.first_seen_round),
                    )
                observed.quality_first_round = _quality_first_round(
                    observed, previous, self._round_number
                )
                if previous is None or float(observed.confidence) >= float(
                    previous.confidence
                ):
                    self._map_items[key] = observed
            return self._changed()

    def apply_packet_snapshot(self, snapshot: ObservationSnapshot) -> int:
        """Take one round's evidence from packet capture; human corrections still win."""
        with self._lock:
            round_number = max(1, min(5, int(snapshot.round_number)))
            self._round_number = round_number
            for raw in snapshot.events:
                key = (int(raw["round_number"]), str(raw["kind"]))
                previous = self._events.get(key)
                if previous is not None and previous.human_locked:
                    continue
                self._events[key] = EventObservation(**copy.deepcopy(dict(raw)))
            captured = {}
            for raw in snapshot.map_items:
                item = MapObservation(**copy.deepcopy(dict(raw)))
                captured[item.key] = item
            for key, previous in list(self._map_items.items()):
                if not previous.human_locked and key not in captured:
                    del self._map_items[key]
            for key, item in captured.items():
                previous = self._map_items.get(key)
                if key in self._suppressed_map_keys or (previous is not None and previous.human_locked):
                    continue
                self._map_items[key] = item
            if self._map_extent_by_round.get(round_number, (None, False, ""))[2] != "human_confirmed":
                self._map_extent_by_round[round_number] = (
                    snapshot.map_rows, bool(snapshot.map_height_exact), snapshot.map_rows_source,
                )
            if self._complete_rounds.get(round_number) != "human_confirmed":
                self._complete_rounds[round_number] = snapshot.completeness_source
            if snapshot.pre_bid_confirmed and self._pre_bid_rounds.get(round_number) != "human_confirmed":
                self._pre_bid_rounds[round_number] = snapshot.pre_bid_source
            revision = self._changed()
            if snapshot.packet_decision is not None:
                self._packet_decision = (revision, round_number, copy.deepcopy(snapshot.packet_decision))
            return revision

    def correct_event(
        self,
        round_number: int,
        kind: str,
        text: str,
        semantics: dict,
    ) -> int:
        if kind not in {"public", "personal"}:
            raise ValueError("kind 必须是 public 或 personal")
        with self._lock:
            self._events[(int(round_number), kind)] = EventObservation(
                round_number=int(round_number),
                kind=kind,
                text=str(text).strip(),
                confidence=1.0,
                semantics=copy.deepcopy(semantics),
                source="human",
                human_locked=True,
            )
            return self._changed()

    def correct_bankroll(self, value: int | None) -> int:
        with self._lock:
            self._bankroll = int(value) if value is not None else None
            self._bankroll_human_locked = True
            return self._changed()

    def upsert_map_item(self, item: MapObservation) -> int:
        corrected = copy.deepcopy(item)
        corrected.source = "human"
        corrected.human_locked = True
        corrected.confidence = 1.0
        with self._lock:
            previous = self._map_items.get(corrected.key)
            if corrected.first_seen_round is None:
                corrected.first_seen_round = (
                    int(previous.first_seen_round)
                    if previous is not None
                    and previous.first_seen_round is not None
                    else self._round_number
                )
            corrected.quality_first_round = _quality_first_round(
                corrected, previous, self._round_number
            )
            self._suppressed_map_keys.discard(corrected.key)
            self._map_items[corrected.key] = corrected
            return self._changed()

    def remove_map_item(self, row: int, column: int) -> int:
        key = (int(row), int(column))
        with self._lock:
            self._map_items.pop(key, None)
            self._suppressed_map_keys.add(key)
            return self._changed()

    def confirm_complete(self, confirmed: bool) -> int:
        with self._lock:
            if confirmed:
                self._complete_rounds[self._round_number] = "human_confirmed"
            else:
                self._complete_rounds.pop(self._round_number, None)
            return self._changed()

    def correct_map_extent(self, rows: int | None, exact: bool) -> int:
        parsed_rows = int(rows) if rows is not None else None
        if parsed_rows is not None and parsed_rows <= 0:
            raise ValueError("地图总行数必须大于 0")
        with self._lock:
            self._map_extent_by_round[self._round_number] = (
                parsed_rows,
                bool(exact and parsed_rows is not None),
                "human_confirmed",
            )
            return self._changed()

    def confirm_pre_bid(self, confirmed: bool) -> int:
        with self._lock:
            if confirmed:
                self._pre_bid_rounds[self._round_number] = "human_confirmed"
            else:
                self._pre_bid_rounds.pop(self._round_number, None)
            return self._changed()

    def apply_automatic_scan_proof(
        self,
        *,
        round_number: int,
        map_rows: int | None,
        complete: bool,
        pre_bid: bool,
    ) -> int:
        """Attach explicit, provisional proof from the visual watch loop."""

        round_number = max(1, min(5, int(round_number)))
        parsed_rows = int(map_rows) if map_rows is not None else None
        if parsed_rows is not None and parsed_rows <= 0:
            raise ValueError("地图总行数必须大于 0")
        with self._lock:
            self._round_number = round_number
            if complete and self._complete_rounds.get(round_number) != "human_confirmed":
                self._complete_rounds[round_number] = "automatic_scroll_scan"
            if pre_bid and self._pre_bid_rounds.get(round_number) != "human_confirmed":
                self._pre_bid_rounds[round_number] = "visual_state_machine"
            previous_extent = self._map_extent_by_round.get(round_number)
            if parsed_rows is not None and (
                previous_extent is None or previous_extent[2] != "human_confirmed"
            ):
                self._map_extent_by_round[round_number] = (
                    parsed_rows,
                    False,
                    "automatic_scroll_estimate",
                )
            return self._changed()

    def snapshot(self) -> ObservationSnapshot:
        with self._lock:
            map_rows, map_height_exact, map_rows_source = (
                self._map_extent_by_round.get(
                    self._round_number, (None, False, "missing")
                )
            )
            return ObservationSnapshot(
                revision=self._revision,
                round_number=self._round_number,
                bankroll=self._bankroll,
                events=tuple(
                    asdict(value)
                    for key, value in sorted(self._events.items())
                    if key[0] <= self._round_number
                ),
                map_items=tuple(
                    asdict(value)
                    for _, value in sorted(self._map_items.items())
                ),
                completeness_confirmed=(
                    self._round_number in self._complete_rounds
                ),
                completeness_source=self._complete_rounds.get(
                    self._round_number, "missing"
                ),
                map_rows=map_rows,
                map_height_exact=map_height_exact,
                map_rows_source=map_rows_source,
                pre_bid_confirmed=(
                    self._round_number in self._pre_bid_rounds
                ),
                pre_bid_source=self._pre_bid_rounds.get(
                    self._round_number, "missing"
                ),
                packet_decision=self._current_packet_decision(),
            )

    def _current_packet_decision(self) -> dict | None:
        held = self._packet_decision
        if held is None or held[0] != self._revision or held[1] != self._round_number:
            return None
        return copy.deepcopy(held[2])

    def preview_snapshot(
        self, round_number: int, events: Iterable[EventObservation]
    ) -> ObservationSnapshot:
        """The snapshot a capture of these events would give, without applying it.

        Used to start inference before the round's map scan ends: the map,
        its completeness and its extent are those of the previous scan, and
        human-locked events still win over new OCR.
        """
        value = max(1, min(5, int(round_number)))
        with self._lock:
            merged = dict(self._events)
            for event in events:
                key = (int(event.round_number), str(event.kind))
                previous = merged.get(key)
                if previous is not None and previous.human_locked:
                    continue
                merged[key] = copy.deepcopy(event)
            previous_round = value - 1
            map_rows, map_height_exact, map_rows_source = (
                self._map_extent_by_round.get(previous_round, (None, False, "missing"))
            )
            return ObservationSnapshot(
                revision=self._revision,
                round_number=value,
                bankroll=self._bankroll,
                events=tuple(
                    asdict(event)
                    for key, event in sorted(merged.items())
                    if key[0] <= value
                ),
                map_items=tuple(
                    asdict(item) for _, item in sorted(self._map_items.items())
                ),
                completeness_confirmed=(
                    previous_round == 0 or previous_round in self._complete_rounds
                ),
                completeness_source=(
                    "no_earlier_scan"
                    if previous_round == 0
                    else self._complete_rounds.get(previous_round, "missing")
                ),
                map_rows=map_rows,
                map_height_exact=map_height_exact,
                map_rows_source=map_rows_source,
                pre_bid_confirmed=False,
                pre_bid_source="preview_before_scan",
            )

    def _zero_frame(self) -> None:
        if self._frame is not None:
            self._frame.fill(0)
            self._frame = None

    def reset(self) -> int:
        with self._lock:
            self._zero_frame()
            self._round_number = 1
            self._bankroll = None
            self._bankroll_human_locked = False
            self._events.clear()
            self._map_items.clear()
            self._suppressed_map_keys.clear()
            self._complete_rounds.clear()
            self._map_extent_by_round.clear()
            self._pre_bid_rounds.clear()
            self._packet_decision = None
            return self._changed()
