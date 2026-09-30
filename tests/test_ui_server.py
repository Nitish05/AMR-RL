"""Operator console server: loopback only, whitelisted commands, bounded bodies,
acknowledgements routed back from the simulation thread."""

import json
import threading
import urllib.request

import pytest

from amr_rl.ui.server import Console, serve


@pytest.fixture()
def console():
    c = Console()
    server = serve(c, port=0)
    c.port = server.server_address[1]
    yield c
    server.shutdown()


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/command", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def test_refuses_non_loopback():
    with pytest.raises(ValueError):
        serve(Console(), host="0.0.0.0", port=0)


def test_state_static_and_images(console):
    console.publish({"schema": "amr_rl.state.v1", "x": 1}, {"screen.png": (b"PNG", "image/png")})
    base = f"http://127.0.0.1:{console.port}"
    assert json.loads(urllib.request.urlopen(base + "/api/state").read())["x"] == 1
    assert urllib.request.urlopen(base + "/api/screen.png").read() == b"PNG"
    assert b"<html" in urllib.request.urlopen(base + "/").read()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(base + "/../../etc/passwd")


def test_commands_are_acknowledged_by_the_sim_thread(console):
    def sim_thread():
        for _ in range(100):
            for cid, cmd in console.drain():
                console.acknowledge(cid, {"accepted": cmd["action"] == "stop", "reason": cmd["action"], "generation": 7})
            threading.Event().wait(0.01)

    t = threading.Thread(target=sim_thread)
    t.start()
    assert _post(console.port, {"action": "stop"}) == {"accepted": True, "reason": "stop", "generation": 7}
    t.join()


def test_unknown_actions_rejected_without_reaching_the_sim(console):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(console.port, {"action": "self_destruct"})
    assert err.value.code == 400 and json.loads(err.value.read())["reason"] == "unknown_action"
    assert console.drain() == []


def test_heartbeat_bypasses_the_queue(console):
    console.heartbeat_fn = lambda: {"accepted": True, "reason": "heartbeat", "generation": 0}
    assert _post(console.port, {"action": "heartbeat"})["accepted"]
    assert console.drain() == []
