"""Record the README demo (docs/demo.gif + docs/demo.mp4) by driving the app in headless Chromium.

    uv run python scripts/record_demo.py                                   # rules brain, no key (dry run)
    DEMO_PROVIDER=openai DEMO_MODEL=gpt-5.4-mini DEMO_API_KEY=sk-... \\
        uv run python scripts/record_demo.py                                # LLM agent with your own key

The app runs in public mode on its own port and a throwaway case DB. The key (if given) is typed into the app's
"Use your own API key" box exactly as a visitor would; it is never passed to the server process, written to disk or
shown on screen (password field). Frames are screenshots at 2x pixel density; each frame gets an explicit duration,
so scrolling plays at a steady human pace and holds last as long as asked, independent of how slow screenshots are.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FPS = 10  # scroll animation and GIF frame rate
SCROLL_SPEED = 380  # CSS px per second while scrolling
HOLD = 5.0  # minimum pause at every point of interest
FINAL_HOLD = 12.0

SCROLLER_JS = """() => {
  let best = null;
  for (const el of document.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section, div')) {
    if (el.closest('[data-testid="stSidebar"]')) continue;
    const s = getComputedStyle(el);
    if (!/(auto|scroll)/.test(s.overflowY) || el.scrollHeight <= el.clientHeight + 4) continue;
    if (!best || el.clientHeight > best.clientHeight) best = el;
  }
  return best || document.scrollingElement;
}"""


class Recorder:
    """Screenshots with explicit durations, encoded by ffmpeg's concat demuxer."""

    def __init__(self, page: Page, out: Path):
        self.page, self.out, self.frames = page, out, []

    def shot(self, duration: float) -> None:
        path = self.out / f"{len(self.frames):05d}.jpg"
        self.page.screenshot(path=path, type="jpeg", quality=92)
        self.frames.append((path, duration))

    def hold(self, seconds: float = HOLD) -> None:
        settle(self.page)
        self.shot(seconds)

    def scroll_y(self) -> tuple[float, float]:
        return self.page.evaluate(f"() => {{ const el = ({SCROLLER_JS})(); return [el.scrollTop, el.scrollHeight - el.clientHeight]; }}")

    def scroll_to(self, target: float, speed: float = SCROLL_SPEED) -> None:
        """Ease-in-out scroll of the main pane, one captured frame per step."""
        start, max_y = self.scroll_y()
        target = max(0.0, min(target, max_y))
        steps = max(1, math.ceil(abs(target - start) / speed * FPS))
        for i in range(1, steps + 1):
            t = i / steps
            y = start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * t))
            self.page.evaluate(f"(y) => {{ ({SCROLLER_JS})().scrollTop = y; }}", y)
            self.shot(1 / FPS)

    def scroll_by(self, dy: float) -> None:
        self.scroll_to(self.scroll_y()[0] + dy)

    def scroll_into_view(self, locator, margin: float = 90) -> None:
        top = locator.evaluate(f"(el) => {{ const s = ({SCROLLER_JS})(); "
                               "return el.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop; }")
        self.scroll_to(top - margin)

    def read_down(self, stop_at: float | None = None, pause: float = 1.6) -> None:
        """Scroll to the bottom (or stop_at) in reading-sized steps with short pauses, like a person reading."""
        viewport = self.page.viewport_size["height"]
        while True:
            y, max_y = self.scroll_y()
            end = max_y if stop_at is None else min(stop_at, max_y)
            if y >= end - 2:
                return
            self.scroll_to(min(end, y + viewport * 0.55))
            if self.scroll_y()[0] < end - 2:  # the caller holds at the end
                self.hold(pause)

    def watch(self, done, max_real: float = 240, show: float = 14.0) -> None:
        """Capture while the agent runs; long runs are sped up so they take about `show` seconds on screen."""
        first, t0 = len(self.frames), time.time()
        while not done():
            if time.time() - t0 > max_real:
                raise TimeoutError("agent run did not finish")
            t = time.time()
            self.shot(0)
            time.sleep(max(0.0, 0.5 - (time.time() - t)))
            self.frames[-1] = (self.frames[-1][0], time.time() - t)
        real = sum(d for _, d in self.frames[first:])
        k = min(1.0, show / real) if real else 1.0
        self.frames[first:] = [(p, d * k) for p, d in self.frames[first:]]

    def encode(self, gif: Path, mp4: Path, gif_width: int) -> None:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        lst = self.out / "frames.txt"
        lines = [f"file '{p}'\nduration {d:.4f}" for p, d in self.frames]
        lst.write_text("\n".join(lines + [f"file '{self.frames[-1][0]}'"]) + "\n")
        src = ["-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
        subprocess.run([ffmpeg, *src, "-vf", f"fps={FPS},scale=1920:-2:flags=lanczos", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
        vf = (f"fps={FPS},scale={gif_width}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle")  # flat UI: ~10 MB
        subprocess.run([ffmpeg, *src, "-vf", vf, "-loop", "0", str(gif)], check=True)


def settle(page: Page, timeout: float = 60) -> None:
    """Wait until Streamlit finished rerunning (no 'Running...' indicator, no stale-element fade)."""
    page.wait_for_timeout(300)
    t0 = time.time()
    while time.time() - t0 < timeout:
        busy = page.evaluate("""() => !!document.querySelector('[data-testid="stStatusWidget"]')
            || !!document.querySelector('[data-stale="true"]')""")
        if not busy:
            break
        page.wait_for_timeout(200)
    page.wait_for_timeout(400)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_app(db: Path) -> tuple[subprocess.Popen, str]:
    port = free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("DEMO_")}  # the key goes in through the UI only
    env |= {"DISPUTE_AGENT_PUBLIC": "1", "DISPUTE_AGENT_DB": str(db)}
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.port", str(port),
                             "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return proc, f"http://127.0.0.1:{port}"
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("streamlit did not start")


def choose(page: Page, label: str, option_prefix: str) -> None:
    page.locator('[data-testid="stSelectbox"]').filter(has_text=label).click()
    page.get_by_role("option").filter(has_text=re.compile("^" + re.escape(option_prefix))).first.click()
    settle(page)


def finished(page: Page):
    pat = re.compile("Done: agent steps|Waiting for a dispute officer|Stopped: the model call failed")
    return lambda: page.get_by_text(pat).count() > 0


def file_case(rec: Recorder, page: Page, customer: str, first: bool = False) -> None:
    if not first:  # the first account is already on screen (landing hold)
        rec.scroll_to(0)
        choose(page, "Customer account", customer)
        rec.hold()  # the account, its transactions and the complaint
    page.get_by_role("button", name="Send complaint").click()
    page.wait_for_timeout(600)
    rec.scroll_into_view(page.locator('[data-testid="stExpander"]').last)
    rec.watch(finished(page))
    if page.get_by_text("Stopped: the model call failed").count():
        raise SystemExit("the model call failed (check the key / model); see the app screenshot in the frames dir")
    rec.hold()  # the full trace
    rec.read_down()
    rec.hold()  # outcome: decision, cited clauses, reply


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", default=ROOT / "docs" / "demo.gif", type=Path)
    ap.add_argument("--mp4", default=ROOT / "docs" / "demo.mp4", type=Path)
    ap.add_argument("--gif-width", default=1280, type=int)
    ap.add_argument("--keep-frames", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("DEMO_API_KEY", "").strip()
    provider, model = os.environ.get("DEMO_PROVIDER", "openai"), os.environ.get("DEMO_MODEL", "")

    tmp = Path(tempfile.mkdtemp(prefix="demo-rec-"))
    proc, url = start_app(tmp / "cases.sqlite")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2, color_scheme="light")
            page.goto(url)
            page.get_by_role("button", name="Send complaint").wait_for(timeout=60_000)
            settle(page)
            if key:  # set up before recording starts; the password field never shows the value anyway
                page.get_by_text("Anthropic" if provider == "anthropic" else "OpenAI", exact=True).click()
                settle(page)
                if model:
                    choose(page, "Model", model)
                page.get_by_label("API key").fill(key)
                page.get_by_role("button", name="Use key").click()
                settle(page)
                page.get_by_text("Using your").wait_for(state="attached", timeout=10_000)  # inside the now-collapsed expander
                page.get_by_text("LLM agent loop").click()
                settle(page)
            rec = Recorder(page, tmp)

            rec.hold(HOLD + 1)  # landing: settings sidebar + first customer
            file_case(rec, page, "C001", first=True)  # duplicate charge -> refund
            file_case(rec, page, "C002")  # monthly subscription -> reject
            file_case(rec, page, "C006")  # 899 EUR -> waits for a human

            page.locator('[data-testid="stSidebarNav"] a').filter(has_text="Review queue").click()
            page.get_by_role("button", name="Submit verdict").wait_for(timeout=30_000)
            rec.hold()  # why it is waiting, evidence, the pre-filled 899 EUR
            page.get_by_label("Note for the audit trail").fill("called the customer, confirmed card fraud")
            rec.hold(2.5)
            page.get_by_role("button", name="Submit verdict").click()
            page.wait_for_timeout(600)
            rec.scroll_into_view(page.locator('[data-testid="stExpander"]').last)
            rec.watch(finished(page))
            rec.hold()
            rec.read_down()
            rec.hold()  # final decision after the officer's verdict

            page.locator('[data-testid="stSidebarNav"] a').filter(has_text="Results").click()
            page.get_by_text("Decision accuracy").first.wait_for(timeout=30_000)
            page.wait_for_timeout(1500)  # charts draw
            rec.hold(HOLD + 2)  # headline tiles + accuracy / cost charts
            rec.read_down(pause=HOLD)
            rec.scroll_to(0, speed=900)
            rec.hold(FINAL_HOLD)
            browser.close()
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        rec.encode(args.gif, args.mp4, args.gif_width)
        total = sum(d for _, d in rec.frames)
        print(f"{len(rec.frames)} frames, {total:.0f} s -> {args.gif} ({args.gif.stat().st_size / 1e6:.1f} MB), "
              f"{args.mp4} ({args.mp4.stat().st_size / 1e6:.1f} MB)")
    finally:
        proc.terminate()
        if args.keep_frames:
            print("frames kept in", tmp)
        else:
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()


if __name__ == "__main__":
    main()
