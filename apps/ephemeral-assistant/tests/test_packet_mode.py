"""Packet-capture input (the default): decoding, model inputs, store, monitor, CLI."""

from __future__ import annotations

import importlib.util
import io
import struct
import unittest
from pathlib import Path

from auction_moment_assistant.cli import build_parser, exact_enabled
from auction_moment_assistant.models import CatalogItem
from auction_moment_assistant.observations import ObservationStore
from auction_moment_assistant.packet_protocol import (
    MATCH_OPENED,
    PERSONAL_EVENT_RESOLVED,
    ROUND_EVENT_OFFERED,
    PacketEvent,
    decode_pcap,
)
from auction_moment_assistant.packet_session import PacketGame, PacketGameError, PacketMonitor, load_codebook

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
EVENT_CATALOG = Path(__file__).resolve().parents[3] / "data" / "v1" / "core" / "event_catalog.csv"


# ---------------------------------------------------------------- wire format
def varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def field(number: int, value) -> bytes:
    if isinstance(value, bytes):
        return varint(number << 3 | 2) + varint(len(value)) + value
    return varint(number << 3) + varint(value)


def item(index, *, quality=None, value=None, cells=()):
    body = field(1, index) + field(2, 900 + index)
    if quality is not None:
        body += field(3, quality)
    for row, column in cells:
        body += field(4, field(1, column) + field(2, row))
    if value is not None:
        body += field(5, value)
    return body


def effect(code, observed=0, items=()):
    return field(1, code) + field(2, observed) + b"".join(field(3, i) for i in items)


def frame(command_hex: str, body: bytes) -> bytes:
    payload = bytes.fromhex(command_hex) + body
    return len(payload).to_bytes(2, "big") + payload


def ethernet(payload: bytes, sequence: int, *, server=True, flags=0x18) -> bytes:
    ports = (8002, 50000) if server else (50000, 8002)
    tcp = struct.pack("!HHIIBBHHH", *ports, sequence, 0, 5 << 4, flags, 65535, 0, 0)
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp) + len(payload), 0, 0, 64, 6, 0,
                     bytes([10, 0, 0, 1] if server else [10, 0, 0, 2]),
                     bytes([10, 0, 0, 2] if server else [10, 0, 0, 1]))
    return b"\x00" * 12 + b"\x08\x00" + ip + tcp + payload


def pcap(packets) -> bytes:
    out = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for number, packet in enumerate(packets):
        out += struct.pack("<IIII", 1_800_000_000 + number, 0, len(packet), len(packet)) + packet
    return out


def match_stream():
    """A two-round match, with frames the assistant must not decode mixed in."""
    players = field(1, field(1, 123456789) + field(2, 1) + field(3, field(2, 70)))
    return b"".join([
        frame("005b680000", players + field(2, 9)),
        frame("005b690000", field(1, 1) + field(2, 1)
              + field(3, effect(402, 0, [item(1, quality=4, cells=[(1, 1)]), item(2, quality=1, cells=[(3, 4)])]))
              + b"".join(field(4, code) for code in (1801, 301, 601)) + field(5, 30)),
        frame("005b6b0000", field(1, 0) + field(2, effect(1101, 5))),
        frame("005b6f0000", field(1, 1) + field(2, field(1, 123456789) + field(2, 5000))),  # ranking
        frame("005b690000", field(1, 2) + field(2, 2) + field(3, effect(1201, 34))
              + b"".join(field(4, code) for code in (1801, 302, 602))),
        frame("005b6b0000", field(1, 0) + field(2, effect(501, 0, [item(3, quality=5, value=1500, cells=[(5, 1)])]))),
        frame("005b700000", field(1, 123456789) + field(2, 9000)),  # settlement
    ])


def events_from_pcap(data: bytes) -> list[PacketEvent]:
    return list(decode_pcap(io.BytesIO(data)))


CATALOG = (
    CatalogItem("w1", "白1", "白", 1, 1, 10),
    CatalogItem("g1", "金1", "金", 1, 2, 409),
    CatalogItem("c1", "彩1", "彩", 1, 1, 1500),
)
CODEBOOK = {
    402: {"effect": "reveal_position_quality_random", "event_template": "随机显示5件藏品的位置与品质",
          "group_dimension": "random", "group_value": "5", "value_scale": "1"},
    1101: {"effect": "highest_quality", "event_template": "显示本局藏品的最高品质",
           "group_dimension": "global", "group_value": "highest_quality", "value_scale": "1"},
    1201: {"effect": "total_occupied_cells", "event_template": "显示本局藏品总占格数",
           "group_dimension": "global", "group_value": "total_occupied_cells", "value_scale": "1"},
    501: {"effect": "reveal_identity_random", "event_template": "随机完整揭示1件藏品",
          "group_dimension": "random", "group_value": "1", "value_scale": "1"},
}


class PacketDecodingTests(unittest.TestCase):
    def test_only_visible_frames_are_decoded_across_split_and_reordered_segments(self):
        stream = match_stream()
        cut = [0, 7, 40, 41, 90, len(stream)]
        chunks = [(cut[i], stream[cut[i]:cut[i + 1]]) for i in range(len(cut) - 1)]
        base = 1000
        packets = [ethernet(b"", base, flags=0x02)]  # handshake, no payload
        order = [0, 2, 1, 3, 1, 4]  # one reordering and one retransmission
        packets += [ethernet(chunks[i][1], base + chunks[i][0]) for i in order]
        packets.append(ethernet(frame("005b690000", field(2, 3)), 1, server=False))  # client side: ignored
        events = events_from_pcap(pcap(packets))
        self.assertEqual([e.kind for e in events],
                         [MATCH_OPENED, ROUND_EVENT_OFFERED, PERSONAL_EVENT_RESOLVED,
                          ROUND_EVENT_OFFERED, PERSONAL_EVENT_RESOLVED])
        self.assertEqual(events[0].payload, {"map_rows": 9})  # players are skipped
        offer = events[1].payload
        self.assertEqual((offer["round"], offer["personal_event_options"]), (1, [1801, 301, 601]))
        self.assertEqual(offer["public_event"]["items"][0],
                         {"index": 1, "quality_code": 4, "cells": [{"column": 1, "row": 1}], "value": None})
        self.assertNotIn("123456789", repr([e.payload for e in events]))

    def test_a_truncated_body_is_reported_not_raised(self):
        events = events_from_pcap(pcap([ethernet(frame("005b690000", b"\x1a\x05\x08"), 1)]))
        self.assertEqual((events[0].kind, events[0].valid), (ROUND_EVENT_OFFERED, False))


class PacketGameTests(unittest.TestCase):
    def game(self, upto=None):
        game = PacketGame(CODEBOOK, CATALOG)
        for event in events_from_pcap(pcap([ethernet(match_stream(), 1)]))[:upto]:
            game.add(event)
        return game

    def test_decision_and_snapshot_follow_the_capture(self):
        game = self.game()
        decision = game.decision(2)
        self.assertEqual((decision["map_rows"], [e["effect"] for e in decision["events"]]),
                         (9, ["reveal_position_quality_random", "highest_quality",
                              "total_occupied_cells", "reveal_identity_random"]))
        self.assertEqual(decision["events"][0]["items"][0], {"index": 1, "quality": "金", "value": None,
                                                             "cells": [[0, 0]]})
        snapshot = game.snapshot(2)
        self.assertEqual((snapshot.map_rows, snapshot.map_height_exact, snapshot.pre_bid_confirmed), (9, True, True))
        self.assertEqual(snapshot.events[1]["semantics"]["observed_quality"], "彩")
        complete = [i for i in snapshot.map_items if i["spatial"] == "complete"]
        self.assertEqual((complete[0]["catalog_id"], complete[0]["first_seen_round"]), ("c1", 2))
        self.assertEqual(snapshot.packet_decision, decision)

    def test_prewarm_snapshot_stops_at_the_public_offer(self):
        game = self.game(upto=4)  # round 2 offered, personal result not in yet
        self.assertFalse(game.ready(2))
        partial = game.snapshot(2, partial=True)
        self.assertFalse(partial.pre_bid_confirmed)
        self.assertEqual([e["kind"] for e in partial.events if e["round_number"] == 2], ["public"])

    def test_a_capture_that_missed_the_match_opening_gives_no_inputs(self):
        game = PacketGame(CODEBOOK, CATALOG)
        for event in events_from_pcap(pcap([ethernet(match_stream(), 1)]))[1:]:
            game.add(event)
        with self.assertRaises(PacketGameError):
            game.snapshot(2)

    def test_public_codebook_covers_all_event_codes(self):
        codebook = load_codebook(EVENT_CATALOG)
        self.assertEqual(len(codebook), 46)
        self.assertEqual(codebook[1801]["effect"], "size_average_value")


class PacketStoreAndMonitorTests(unittest.TestCase):
    def test_store_keeps_the_decision_only_while_unedited(self):
        store = ObservationStore()
        game = PacketGameTests.game(self)
        store.apply_packet_snapshot(game.snapshot(2))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.packet_decision["round"], 2)
        self.assertEqual((snapshot.map_rows, snapshot.map_height_exact), (9, True))
        store.correct_event(2, "public", "显示本局藏品总占格数35", {"parsed": True, "effect": "total_occupied_cells",
                                                                  "observed_cells": 35})
        self.assertIsNone(store.snapshot().packet_decision)

    def test_monitor_applies_ready_rounds_and_prewarms_on_offers(self):
        events = events_from_pcap(pcap([ethernet(match_stream(), 1)]))

        class Source:
            def __init__(self, listener):
                self.listener = listener

            def start(self):
                for event in events:
                    self.listener(event)

            def stop(self):
                pass

        store, previews = ObservationStore(), []
        monitor = PacketMonitor(Source, PacketGame(CODEBOOK, CATALOG), store, on_preview=previews.append)
        monitor.start()
        kinds = [update.kind for update in monitor.drain()]
        self.assertEqual(kinds.count("scan_finished"), 2)
        self.assertIn("session_started", kinds)
        self.assertEqual([p.round_number for p in previews], [1, 2])
        self.assertEqual(store.snapshot().round_number, 2)
        self.assertIsNotNone(store.snapshot().packet_decision)


class PacketDefaultsTests(unittest.TestCase):
    def test_packet_input_and_exact_model_are_the_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual((args.input, args.protocol_port, args.packet_interface), ("packet", 8002, "wlan0"))
        self.assertTrue(exact_enabled(args))
        self.assertFalse(exact_enabled(build_parser().parse_args(["--v6-only"])))
        self.assertFalse(exact_enabled(build_parser().parse_args(["--experimental-v2-fusion"])))
        self.assertEqual(build_parser().parse_args(["--input", "ocr"]).input, "ocr")


@unittest.skipUnless(HAS_SCIPY, "exact sampler needs scipy")
class PacketEvidenceTests(unittest.TestCase):
    def test_decision_evidence_uses_every_draw(self):
        from auction_moment_assistant.exact_evidence import evidence_from_decision

        result = evidence_from_decision(PacketGameTests.game(self).decision(2), CATALOG)
        self.assertTrue(result.usable, result.unsupported)
        ev = result.evidence
        self.assertEqual((ev.highest, ev.total_cells, ev.map_rows), (4, 34, 9))
        self.assertEqual(dict(ev.revealed_identities), {2: 1})
        self.assertEqual(dict(ev.revealed_qualities), {3: 1, 0: 1})

    def test_unknown_or_conflicting_effects_block_the_mode(self):
        from auction_moment_assistant.exact_evidence import evidence_from_decision

        def event(effect, **fields):
            return {"round": 1, "scope": "public", "effect": effect, "group_value": "金", "observed_value": 3,
                    "value_scale": 1, "reported_item_count": 0, "reported_cell_count": 0, "items": [], **fields}

        unknown = evidence_from_decision({"map_rows": 9, "events": [event("total_item_count")]}, CATALOG)
        self.assertFalse(unknown.usable)
        conflict = evidence_from_decision({"map_rows": 9, "events": [
            event("quality_item_count"), event("reveal_position_by_quality", reported_item_count=4)]}, CATALOG)
        self.assertTrue(any(r.startswith("conflicting_clues") for r in conflict.unsupported))

    def test_predictor_reads_the_decision_when_the_snapshot_carries_one(self):
        import types

        from auction_moment_assistant.latest_model import LatestV6V2Predictor

        predictor = LatestV6V2Predictor.__new__(LatestV6V2Predictor)
        predictor.exact_catalog = CATALOG
        predictor.adapter = types.SimpleNamespace(minimum_ocr_confidence=0.85)
        import dataclasses

        snapshot = PacketGameTests.game(self).snapshot(2)
        self.assertEqual(predictor._exact_evidence(snapshot).evidence.revealed_qualities, {3: 1, 0: 1})
        # the decision, not the snapshot, is what gets read
        other = dataclasses.replace(snapshot, packet_decision={"map_rows": 7, "events": []})
        self.assertEqual(predictor._exact_evidence(other).evidence.map_rows, 7)
        screen = dataclasses.replace(snapshot, packet_decision=None)
        self.assertEqual(predictor._exact_evidence(screen).evidence.map_rows, 9)


if __name__ == "__main__":
    unittest.main()
