"""Sources of visible auction events: live passive capture, or a PCAP file.

``AdbPacketCapture`` runs tcpdump on a rooted emulator through ADB, reads
its PCAP stream from ``adb exec-out`` and decodes it in memory; it never
sends taps, keys or bids, and writes nothing on the host. The only file it
touches is a pid file on the device, removed when capture stops, so the
right tcpdump can be stopped. ``PcapFilePacketSource`` replays a capture
file the same way.
"""

from __future__ import annotations

import subprocess
import threading
import uuid
from pathlib import Path
from typing import Callable

from .packet_protocol import (
    DEFAULT_PROTOCOL_PORT,
    PacketDecoder,
    PacketEvent,
    PacketProtocolError,
    PcapStreamReader,
)


class PacketCaptureError(RuntimeError):
    pass


EventListener = Callable[[PacketEvent], None]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _ReaderSource:
    def __init__(self, on_event: EventListener, protocol_port: int) -> None:
        self.on_event = on_event
        self.protocol_port = int(protocol_port)
        self.status = "created"
        self.last_error: str | None = None
        self.event_count = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()

    def _consume(self, stream) -> None:
        decoder = PacketDecoder(self.protocol_port)
        try:
            reader = PcapStreamReader(stream)
            self.status = "running"
            self._ready.set()
            for record in reader.records():
                if self._stop.is_set():
                    break
                for event in decoder.feed(record):
                    self.event_count += 1
                    self.on_event(event)
            if not self._stop.is_set():
                self.status = "finished"
        except (PacketProtocolError, OSError, ValueError) as exc:
            if not self._stop.is_set():
                self.last_error = str(exc)
                self.status = "failed"
        finally:
            self._ready.set()

    def health(self) -> dict:
        return {"status": self.status, "events": self.event_count, "error": self.last_error}


class PcapFilePacketSource(_ReaderSource):
    """Replay a classic PCAP file (Ethernet, tcp/<port>) in a background thread."""

    def __init__(self, path: Path, on_event: EventListener, *, protocol_port: int = DEFAULT_PROTOCOL_PORT) -> None:
        super().__init__(on_event, protocol_port)
        self.path = Path(path)

    def start(self) -> None:
        if not self.path.is_file():
            raise PacketCaptureError(f"PCAP file not found: {self.path}")

        def run() -> None:
            with self.path.open("rb") as stream:
                self._consume(stream)

        self._thread = threading.Thread(target=run, name="auction-pcap-replay", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class AdbPacketCapture(_ReaderSource):
    """Passive tcpdump on a rooted emulator, decoded in memory on the host."""

    def __init__(
        self,
        on_event: EventListener,
        *,
        adb: str = "adb",
        serial: str = "",
        interface: str = "wlan0",
        protocol_port: int = DEFAULT_PROTOCOL_PORT,
    ) -> None:
        super().__init__(on_event, protocol_port)
        self.adb = str(adb)
        self.serial = str(serial)
        self.interface = str(interface)
        if not self.interface.replace("_", "").replace("-", "").isalnum():
            raise ValueError("capture interface must be a plain interface name")
        self.pid_file = f"/data/local/tmp/auction-assistant-{uuid.uuid4().hex}.pid"
        self.process: subprocess.Popen | None = None

    def _command(self, *arguments: str) -> list[str]:
        return [self.adb, *(("-s", self.serial) if self.serial else ()), *arguments]

    def _run(self, *arguments: str, timeout: float = 10.0) -> str:
        completed = subprocess.run(
            self._command(*arguments), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False, creationflags=NO_WINDOW,
        )
        return completed.stdout.decode("utf-8", errors="replace")

    def start(self, timeout: float = 8.0) -> None:
        try:
            if "uid=0(root)" not in self._run("shell", "id"):
                raise PacketCaptureError("抓包需要 root ADB（模拟器内运行 tcpdump）；也可改用 --input ocr")
            if not self._run("shell", "command -v tcpdump").strip():
                raise PacketCaptureError("模拟器内没有 tcpdump；也可改用 --input ocr")
        except (OSError, subprocess.SubprocessError) as exc:
            raise PacketCaptureError(f"无法通过 ADB 启动抓包：{exc}") from exc
        remote = (
            f"echo $$ > {self.pid_file}; "
            f"exec tcpdump -i {self.interface} -s 0 -U -n -w - 'tcp port {self.protocol_port}' 2>/dev/null"
        )
        self.process = subprocess.Popen(
            self._command("exec-out", "sh", "-c", remote),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0, creationflags=NO_WINDOW,
        )
        self.status = "starting"
        self._thread = threading.Thread(
            target=self._consume, args=(self.process.stdout,), name="auction-packet-capture", daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout) or self.status == "failed":
            error = self.last_error or "tcpdump 没有输出 PCAP 头"
            self.stop()
            raise PacketCaptureError(f"抓包启动失败：{error}；也可改用 --input ocr")

    def stop(self) -> None:
        self._stop.set()
        try:
            pid = self._run("shell", f"cat {self.pid_file} 2>/dev/null", timeout=2.0).strip()
            if pid.isdigit():
                command_line = self._run("shell", f"tr '\\0' ' ' < /proc/{pid}/cmdline 2>/dev/null", timeout=2.0)
                if "tcpdump" in command_line and f"port {self.protocol_port}" in command_line:
                    self._run("shell", f"kill -2 {pid}", timeout=2.0)
            self._run("shell", f"rm -f {self.pid_file}", timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            pass
        if self.process is not None:
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.status != "failed":
            self.status = "stopped"
