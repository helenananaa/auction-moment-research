"""Passive decoding of the auction frames that are visible before bidding.

The game client talks to its server over tcp/8002 with length-prefixed
frames whose body is protobuf. Only three server frames are decoded here:

* the match opening, for the map height (player entries are skipped);
* the round's public event offer, with the items the event reveals;
* the player's own resolved personal event, with the items it reveals.

Every other frame is dropped before decoding, so settlement, rankings, bids
and player identifiers never enter the assistant. Nothing is written to disk.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Iterator


DEFAULT_PROTOCOL_PORT = 8002
MAX_APPLICATION_FRAME_BYTES = 64 * 1024
MAX_STREAM_BUFFER_BYTES = 2 * 1024 * 1024
MATCH_OPENED = "match_opened"
ROUND_EVENT_OFFERED = "round_event_offered"
PERSONAL_EVENT_RESOLVED = "personal_event_resolved"


class PacketProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class PcapRecord:
    timestamp: float
    payload: bytes


class PcapStreamReader:
    """Classic PCAP records from a file or a live ``tcpdump -w -`` pipe."""

    _MAGIC = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000_000.0),
        b"\xa1\xb2\xc3\xd4": (">", 1_000_000.0),
        b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000.0),
        b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000.0),
    }

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        header = self._read_exact(24)
        if header[:4] not in self._MAGIC:
            raise PacketProtocolError(f"unsupported PCAP magic: {header[:4].hex()}")
        self.endian, self.timestamp_divisor = self._MAGIC[header[:4]]
        *_, self.snaplen, link_type = struct.unpack(self.endian + "HHIIII", header[4:])
        if link_type != 1:
            raise PacketProtocolError(f"expected Ethernet link type 1, got {link_type}")

    def _read_exact(self, size: int, *, allow_eof: bool = False) -> bytes:
        output = bytearray()
        while len(output) < size:
            chunk = self.source.read(size - len(output))
            if not chunk:
                if allow_eof and not output:
                    return b""
                raise PacketProtocolError(f"PCAP stream ended after {len(output)}/{size} bytes")
            output.extend(chunk)
        return bytes(output)

    def records(self) -> Iterator[PcapRecord]:
        while True:
            header = self._read_exact(16, allow_eof=True)
            if not header:
                return
            seconds, fraction, captured, _original = struct.unpack(self.endian + "IIII", header)
            if captured > max(self.snaplen, MAX_STREAM_BUFFER_BYTES):
                raise PacketProtocolError(f"PCAP record length {captured} exceeds safe limit")
            yield PcapRecord(seconds + fraction / self.timestamp_divisor, self._read_exact(captured))


@dataclass(frozen=True)
class TcpSegment:
    source: tuple[bytes, int]
    destination: tuple[bytes, int]
    sequence: int
    flags: int
    payload: bytes


def parse_tcp_segment(record: PcapRecord) -> TcpSegment | None:
    """The TCP segment of an Ethernet/IPv4 frame, or None for anything else."""
    packet = record.payload
    if len(packet) < 14:
        return None
    ethernet_type = struct.unpack_from("!H", packet, 12)[0]
    ip_offset = 14
    if ethernet_type == 0x8100:
        if len(packet) < 18:
            return None
        ethernet_type = struct.unpack_from("!H", packet, 16)[0]
        ip_offset = 18
    if ethernet_type != 0x0800 or len(packet) < ip_offset + 20:
        return None
    version_ihl = packet[ip_offset]
    ip_header_length = (version_ihl & 0x0F) * 4
    if version_ihl >> 4 != 4 or ip_header_length < 20 or packet[ip_offset + 9] != 6:
        return None
    if struct.unpack_from("!H", packet, ip_offset + 6)[0] & 0x3FFF:
        return None  # fragments
    ip_end = min(len(packet), ip_offset + struct.unpack_from("!H", packet, ip_offset + 2)[0])
    tcp_offset = ip_offset + ip_header_length
    if ip_end < tcp_offset + 20:
        return None
    source_port, destination_port, sequence = struct.unpack_from("!HHI", packet, tcp_offset)
    tcp_header_length = (packet[tcp_offset + 12] >> 4) * 4
    if tcp_header_length < 20 or ip_end < tcp_offset + tcp_header_length:
        return None
    return TcpSegment(
        source=(packet[ip_offset + 12:ip_offset + 16], source_port),
        destination=(packet[ip_offset + 16:ip_offset + 20], destination_port),
        sequence=sequence,
        flags=packet[tcp_offset + 13],
        payload=packet[tcp_offset + tcp_header_length:ip_end],
    )


@dataclass(frozen=True)
class ApplicationFrame:
    server_to_client: bool
    command: bytes
    body: bytes


class _Stream:
    """One direction of a TCP connection, reassembled into application frames."""

    def __init__(self, server_to_client: bool) -> None:
        self.server_to_client = server_to_client
        self.expected: int | None = None
        self.pending: dict[int, bytes] = {}
        self.buffer = bytearray()

    def feed(self, segment: TcpSegment) -> list[ApplicationFrame]:
        if not segment.payload:
            return []
        if self.expected is None:
            self.expected = segment.sequence
        delta = ((segment.sequence - (self.expected & 0xFFFFFFFF) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
        payload, start = segment.payload, self.expected + delta
        if delta < 0:  # retransmission overlapping data already consumed
            if -delta >= len(payload):
                return []
            payload, start = payload[-delta:], self.expected
        if start > self.expected:
            if len(payload) > len(self.pending.get(start, b"")):
                self.pending[start] = payload
            if sum(map(len, self.pending.values())) > MAX_STREAM_BUFFER_BYTES:
                raise PacketProtocolError("out-of-order TCP buffer exceeded limit")
            return []
        frames = self._append(payload)
        self.expected += len(payload)
        while self.expected in self.pending:
            chunk = self.pending.pop(self.expected)
            frames.extend(self._append(chunk))
            self.expected += len(chunk)
        return frames

    def _append(self, payload: bytes) -> list[ApplicationFrame]:
        self.buffer.extend(payload)
        if len(self.buffer) > MAX_STREAM_BUFFER_BYTES:
            raise PacketProtocolError("application stream buffer exceeded limit")
        frames = []
        while len(self.buffer) >= 2:
            length = int.from_bytes(self.buffer[:2], "big")
            if length < 5 or length > MAX_APPLICATION_FRAME_BYTES:
                raise PacketProtocolError(f"invalid application frame length {length}")
            if len(self.buffer) < 2 + length:
                break
            frame = bytes(self.buffer[2:2 + length])
            del self.buffer[:2 + length]
            frames.append(ApplicationFrame(self.server_to_client, frame[:5], frame[5:]))
        return frames


class FrameReassembler:
    """Application frames of every tcp/<port> stream in capture order."""

    def __init__(self, protocol_port: int = DEFAULT_PROTOCOL_PORT) -> None:
        self.protocol_port = int(protocol_port)
        self._streams: dict[tuple, _Stream] = {}

    def feed(self, segment: TcpSegment) -> list[ApplicationFrame]:
        if self.protocol_port not in (segment.source[1], segment.destination[1]):
            return []
        key = (segment.source, segment.destination)
        stream = self._streams.get(key)
        if stream is None:
            stream = self._streams[key] = _Stream(segment.source[1] == self.protocol_port)
        frames = stream.feed(segment)
        if segment.flags & 0x05:  # FIN or RST ends the stream
            self._streams.pop(key, None)
        return frames


def _varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while offset < len(payload) and shift <= 63:
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise PacketProtocolError("truncated or oversized varint")


def _fields(payload: bytes) -> list[tuple[int, int, int | bytes]]:
    fields = []
    offset = 0
    while offset < len(payload):
        key, offset = _varint(payload, offset)
        number, wire_type = key >> 3, key & 0x07
        if number <= 0:
            raise PacketProtocolError("protobuf field number must be positive")
        if wire_type == 0:
            value, offset = _varint(payload, offset)
        elif wire_type == 1:
            if offset + 8 > len(payload):
                raise PacketProtocolError("truncated fixed64")
            value, offset = struct.unpack_from("<Q", payload, offset)[0], offset + 8
        elif wire_type == 2:
            size, offset = _varint(payload, offset)
            if offset + size > len(payload):
                raise PacketProtocolError("truncated length-delimited field")
            value, offset = payload[offset:offset + size], offset + size
        elif wire_type == 5:
            if offset + 4 > len(payload):
                raise PacketProtocolError("truncated fixed32")
            value, offset = struct.unpack_from("<I", payload, offset)[0], offset + 4
        else:
            raise PacketProtocolError(f"unsupported protobuf wire type {wire_type}")
        fields.append((number, wire_type, value))
        if len(fields) > 100_000:
            raise PacketProtocolError("protobuf field count exceeded limit")
    return fields


def _ints(fields, number: int) -> list[int]:
    return [int(value) for n, wire, value in fields if n == number and wire == 0]


def _int(fields, number: int, default: int | None = None) -> int | None:
    values = _ints(fields, number)
    return values[0] if values else default


def _messages(fields, number: int) -> list[list]:
    return [_fields(bytes(value)) for n, wire, value in fields if n == number and wire == 2]


def _item(fields) -> dict:
    cells = []
    for cell in _messages(fields, 4):
        column, row = _int(cell, 1), _int(cell, 2)
        if column is not None and row is not None:
            cells.append({"column": column, "row": row})
    return {"index": _int(fields, 1), "quality_code": _int(fields, 3), "cells": cells, "value": _int(fields, 5)}


def _effect(fields) -> dict:
    return {
        "code": _int(fields, 1),
        "observed_value": _int(fields, 2),
        "items": [_item(item) for item in _messages(fields, 3)],
    }


def _match_opened(body: bytes) -> dict:
    # Field 1 holds the players; only the map height is read.
    return {"map_rows": _int(_fields(body), 2)}


def _round_event_offered(body: bytes) -> dict:
    fields = _fields(body)
    public = _messages(fields, 3)
    return {
        "round": _int(fields, 2),
        "public_event": _effect(public[0] if public else []),
        "personal_event_options": _ints(fields, 4),
    }


def _personal_event_resolved(body: bytes) -> dict:
    fields = _fields(body)
    result = _messages(fields, 2)
    effect = _effect(result[0] if result else [])
    return {
        "status": _int(fields, 1, 0),
        "event_code": effect["code"],
        "observed_value": effect["observed_value"],
        "items": effect["items"],
    }


VISIBLE_FRAMES = {
    bytes.fromhex("005b680000"): (MATCH_OPENED, _match_opened),
    bytes.fromhex("005b690000"): (ROUND_EVENT_OFFERED, _round_event_offered),
    bytes.fromhex("005b6b0000"): (PERSONAL_EVENT_RESOLVED, _personal_event_resolved),
}


@dataclass(frozen=True)
class PacketEvent:
    sequence: int
    kind: str
    payload: dict
    valid: bool = True
    error: str | None = None


class PacketDecoder:
    """Turn PCAP records into visible auction events, numbered in capture order."""

    def __init__(self, protocol_port: int = DEFAULT_PROTOCOL_PORT) -> None:
        self.reassembler = FrameReassembler(protocol_port)
        self.sequence = 0
        self.frames = 0

    def feed_frame(self, frame: ApplicationFrame) -> PacketEvent | None:
        self.frames += 1
        definition = VISIBLE_FRAMES.get(frame.command) if frame.server_to_client else None
        if definition is None:
            return None
        kind, decode = definition
        self.sequence += 1
        try:
            return PacketEvent(self.sequence, kind, decode(frame.body))
        except PacketProtocolError as exc:
            return PacketEvent(self.sequence, kind, {}, valid=False, error=str(exc))

    def feed(self, record: PcapRecord) -> list[PacketEvent]:
        segment = parse_tcp_segment(record)
        if segment is None:
            return []
        events = (self.feed_frame(frame) for frame in self.reassembler.feed(segment))
        return [event for event in events if event is not None]


def decode_pcap(source: BinaryIO, protocol_port: int = DEFAULT_PROTOCOL_PORT) -> Iterable[PacketEvent]:
    decoder = PacketDecoder(protocol_port)
    for record in PcapStreamReader(source).records():
        yield from decoder.feed(record)
