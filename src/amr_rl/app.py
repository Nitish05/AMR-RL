"""Interactive simulated AMR with the operator console.

    python -m amr_rl.app --world arena [--map DIR] [--memory PATH] [--port 8770]

Simulation only (Genesis World, lockstep). Nothing here connects to hardware.
Autonomy starts DISABLED on every launch, including when memory is restored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .robot.spec import PROJECT_ROOT


def encode(img, ext=".jpg", quality=85):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    ok, data = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), params)
    return data.tobytes() if ok else b""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="arena")
    parser.add_argument("--consequences", default=None, help="name of configs/consequences/<name>.yaml (e.g. standard)")
    parser.add_argument("--map", default=None, help="saved VSLAM map directory to relocalize against")
    parser.add_argument("--memory", default=str(PROJECT_ROOT / "work" / "memory" / "pip.sqlite"))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--realtime", type=float, default=1.0, help="max sim/wall speed ratio (0 = unpaced)")
    parser.add_argument("--no-heartbeat", action="store_true", help="headless use only; disables heartbeat check")
    args = parser.parse_args(argv)

    import yaml

    from .control.supervisor import SupervisorConfig
    from .runtime.robot import RuntimeConfig
    from .sim.harness import Session
    from .ui.server import Console, serve

    run_dir = Path(args.run_dir or PROJECT_ROOT / "work" / "app-runs" / time.strftime("%Y%m%d-%H%M%S"))
    consequences = None
    if args.consequences:
        consequences = yaml.safe_load((PROJECT_ROOT / "configs" / "consequences" / f"{args.consequences}.yaml")
                                      .read_text())
    config = RuntimeConfig(supervisor=SupervisorConfig(require_heartbeat=not args.no_heartbeat))
    session = Session(args.world, run_dir=run_dir, memory_path=args.memory, config=config,
                      consequences=consequences, map_dir=args.map)
    console = Console()
    console.heartbeat_fn = lambda: session.runtime.supervisor.heartbeat().to_dict()
    server = serve(console, port=args.port)
    print(f"AMR-RL console: http://127.0.0.1:{args.port}/  (simulation only; autonomy starts disabled)", flush=True)
    last_images = 0.0
    wall0, sim0 = time.time(), session.now
    try:
        while True:
            for cid, command in console.drain():
                command = dict(command)
                command["_cid"] = cid
                session.runtime.command(command)
            session.control_step()
            _acknowledge(console, session.runtime)
            now = time.time()
            if now - last_images > 0.25:
                last_images = now
                rt = session.runtime
                images = {"camera.jpg": (encode(rt.last_frame.rgb), "image/jpeg"),
                          "screen.png": (encode(rt.screen, ".png"), "image/png")}
                map_img, meta = rt.map_image()
                images["map.png"] = (encode(map_img, ".png"), "image/png")
                inspect = session.world.render_inspection()
                if inspect is not None:
                    images["inspect.jpg"] = (encode(inspect), "image/jpeg")
                state = dict(rt.last_state)
                state["map_meta"] = meta
                state["run_dir"] = str(run_dir)
                console.publish(state, images)
            if args.realtime > 0:
                ahead = (session.now - sim0) / args.realtime - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.2))
    except KeyboardInterrupt:
        pass
    finally:
        summary = session.save_summary()
        server.shutdown()
        session.close()
        print(json.dumps({"run_dir": str(run_dir), "sim_seconds": summary["sim_seconds"],
                          "memory": summary["memory"]}, default=str))


def _acknowledge(console, runtime):
    for ack in runtime.acks:
        cid = ack.get("_cid")
        if cid is not None and not ack.get("_sent"):
            ack["_sent"] = True
            console.acknowledge(cid, {k: v for k, v in ack.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
