"""Loopback-only operator console server (stdlib HTTP; no extra dependencies).

Threading model: the simulation loop (Genesis) runs on the main thread and
publishes immutable snapshots (state JSON + encoded images) into ``Console``.
HTTP handler threads only read those snapshots and enqueue operator commands;
they never touch Genesis or the runtime directly. Commands are applied by the
simulation thread at the next control tick, where Stop wins the batch.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import read_static

MAX_BODY = 4096
ALLOWED = {"stop", "heartbeat", "manual", "enable_autonomy", "disable_autonomy", "goal", "reset_memory"}


class Console:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = b"{}"
        self.images = {}
        self.commands: queue.Queue = queue.Queue(maxsize=256)
        self.acks: dict[int, dict] = {}
        self.ack_events: dict[int, threading.Event] = {}
        self.counter = 0
        # Heartbeats only record operator liveness (a wall-clock timestamp); they are
        # applied immediately so a slow simulation step cannot starve them.
        self.heartbeat_fn = None

    def publish(self, state: dict, images: dict[str, tuple[bytes, str]]):
        body = json.dumps(state, default=_default).encode()
        with self.lock:
            self.state = body
            self.images.update(images)

    def submit(self, command: dict, timeout=2.0) -> dict:
        if command.get("action") == "heartbeat" and self.heartbeat_fn is not None:
            return self.heartbeat_fn()
        with self.lock:
            self.counter += 1
            cid = self.counter
            event = threading.Event()
            self.ack_events[cid] = event
        try:
            self.commands.put_nowait((cid, command))
        except queue.Full:
            return {"accepted": False, "reason": "command_queue_full", "generation": None}
        if command.get("action") == "heartbeat":
            timeout = 0.5
        event.wait(timeout)
        with self.lock:
            self.ack_events.pop(cid, None)
            return self.acks.pop(cid, {"accepted": False, "reason": "no_acknowledgement_yet", "generation": None})

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out

    def acknowledge(self, cid, ack):
        with self.lock:
            self.acks[cid] = ack
            event = self.ack_events.get(cid)
        if event is not None:
            event.set()


def _default(value):
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    return str(value)


def make_handler(console: Console):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AMR-RL-console/1"

        def log_message(self, *_):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                with console.lock:
                    body = console.state
                return self._send(200, body, "application/json")
            if path.startswith("/api/"):
                name = path[len("/api/"):]
                with console.lock:
                    item = console.images.get(name)
                if item is None:
                    return self._send(404, b"not available", "text/plain")
                return self._send(200, item[0], item[1])
            asset = read_static(path)
            if asset is None:
                return self._send(404, b"not found", "text/plain")
            return self._send(200, *asset)

        def do_POST(self):  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/command":
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._send(413, b"bad size", "text/plain")
            try:
                command = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                return self._send(400, b"bad json", "text/plain")
            if not isinstance(command, dict) or command.get("action") not in ALLOWED:
                return self._send(400, json.dumps({"accepted": False, "reason": "unknown_action"}).encode(),
                                  "application/json")
            ack = console.submit(command)
            return self._send(200, json.dumps(ack, default=_default).encode(), "application/json")

    return Handler


def serve(console: Console, host="127.0.0.1", port=8770):
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("The operator console binds to loopback only")
    server = ThreadingHTTPServer((host, port), make_handler(console))
    thread = threading.Thread(target=server.serve_forever, name="console-http", daemon=True)
    thread.start()
    return server


def wait_forever():  # pragma: no cover - interactive helper
    while True:
        time.sleep(3600)
