"""Films of a running world, rendered as the simulation goes (render-only Isaac worker, render.Renderer).

A Film hooks into the world's per-tick recorders: every 1/states_per_s of simulated time it renders the chosen views
of the present state (rigid poses and the paste meshes) and composes one film frame: the views side by side under a
band that names each, and below them a band with the experiment and who acts (assistance), the simulated time, the
playback speed, the operator's running note and the evaluator's latest reading. Nothing is drawn over the rendered
pictures. The frames are encoded at `fps` (states_per_s 12 shown at 24 fps is twice real time, and the band says so).
A film is never spliced: one Film covers one continuous run.
"""

import io
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np

BAND = (13, 21, 30)


def _font(size):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose(pngs, labels, lines, width):
    """One film frame: the views side by side, each under a band naming it, and a band of text lines below."""
    from PIL import Image, ImageDraw
    ims = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    W, H = sum(i.width for i in ims), max(i.height for i in ims)
    head, f_big, f_small = max(28, W//50), _font(max(15, W//80)), _font(max(13, W//100))
    # every line wrapped to the frame's width (the assistance label ran off the right edge)
    measure = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    rows = []
    for k, line in enumerate(lines):
        font = f_big if k == 0 else f_small
        cur = ""
        for word in str(line).split(" "):
            trial = (cur + " " + word).strip()
            if cur and measure.textlength(trial, font=font) > W - 28:
                rows.append((k, cur))
                cur = word
            else:
                cur = trial
        rows.append((k, cur))
    # at most two rows a line, and room for two whatever the text: every frame of a film the same height (ffmpeg)
    kept = []
    for k in range(len(lines)):
        mine = [r for r in rows if r[0] == k]
        if len(mine) > 2:
            mine = mine[:2]
            mine[1] = (k, mine[1][1].rstrip() + " ...")
        kept += mine
    rows = kept
    foot = 14 + 2*(f_big.size + 8) + 2*(len(lines) - 1)*(f_small.size + 7)
    canvas = Image.new("RGB", (W, head + H + foot), BAND)
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, label in zip(ims, labels):
        canvas.paste(im, (x, head))
        d.text((x + 14, (head - f_big.size)//2 - 1), label, fill=(255, 255, 255), font=f_big)
        x += im.width
    y = head + H + 8
    for k, line in rows:
        d.text((14, y), line, fill=(255, 255, 255) if k == 0 else (200, 208, 216), font=f_big if k == 0 else f_small)
        y += (f_big.size + 8) if k == 0 else (f_small.size + 7)
    return canvas


class Film:
    def __init__(self, env, out, cameras, *, title, assistance, views=None, labels=None, pick=None, states_per_s=12, fps=24,
                 width=960, height=540, samples=12, eval_every_s=2.0, work=None):
        from .render import export, Renderer
        self.env, self.world = env, env.world
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cache = Path(work or (Path.home()/".cache"/"gar"/"concrete-finishing-video"))
        cache.mkdir(parents=True, exist_ok=True)
        self.frames_dir = Path(tempfile.mkdtemp(prefix=self.out.stem + "-", dir=cache))
        self.cameras, self.pick, self.labels = list(cameras), pick, labels or {}
        self.title, self.assistance = title, assistance
        self.dt, self.fps = 1.0/states_per_s, fps
        self.width, self.height = width, height
        self.note = ""
        self.written, self.next_t, self.times = 0, self.world.sim_time, []
        self.eval_every, self.next_eval, self.eval_line = eval_every_s, self.world.sim_time, ""
        self.render_s = 0.0
        scene, self.meta = export(self.world, self.frames_dir/"render", views)
        self._scene, self._samples = scene, samples
        from .render import worker_log_size
        self._log_start = worker_log_size(self.frames_dir/"render")
        self.renderer = Renderer(self.frames_dir/"render", scene, width=width, height=height, samples=samples)
        self.restarts, self.frames_held, self.memory_restarts = 0, 0, 0
        self._mesh_cache = {}                             # what this film's worker already has (render.encode_meshes)
        self.world.paste_recorders.append(self)
        self.closed = False
        self.failure = None

    def evaluation_line(self):
        ev = self.env.evaluation()
        crit = ev["criteria"]
        bad = [k for k, c in crit.items() if not c["passed"]]
        return (f"evaluator: score {ev['score']:.2f} ({sum(c['passed'] for c in crit.values())}/{len(crit)} criteria)"
                + (f"; not yet: {', '.join(bad[:5])}" if bad else "; every criterion passed"))

    def __call__(self, world):
        if world.sim_time + 1e-9 < self.next_t:
            return
        self.next_t += self.dt
        self.capture()

    def capture(self):
        if getattr(self, "failure", None):
            raise RuntimeError(self.failure)
        try:
            self._capture()
        except Exception as exc:
            # A demonstration may catch a failed retreat. Latch a missing frame so
            # that it still cannot publish a canonical complete film afterwards.
            self.failure = f"frame capture failed at tick {self.world.tick}: {type(exc).__name__}: {exc}"
            raise

    def _capture(self):
        from .render import frame
        w = self.world
        if w.sim_time + 1e-9 >= self.next_eval:
            self.eval_line = self.evaluation_line()
            self.next_eval = w.sim_time + self.eval_every
        cams = list(self.pick(w.sim_time, self.note)) if self.pick else self.cameras
        t0 = time.perf_counter()
        imgs = None
        from .render import Renderer, worker_log_size, worker_memory_errors
        attempt, memory_tries = 0, 0
        while attempt < 5:
            try:
                imgs = self.renderer.observe(cams, frame(w, self._mesh_cache), w.tick)
                # a worker that met a GPU memory failure may draw a partial scene: it is replaced and the frame asked
                # again (checked on its first frame and every tenth); the GPU is waited for, up to 8 times a minute
                if (self.written == 0 or self.written % 10 == 0) and worker_memory_errors(self.frames_dir/"render", self._log_start):
                    imgs = None
                    memory_tries += 1
                    if memory_tries > 8:
                        raise RuntimeError("film: the render worker met GPU memory failures on 8 starts; no film on a "
                                           "partial scene")
                    print("film: the render worker logged GPU memory failures; waiting 60 s and replacing it", flush=True)
                    self.renderer.close()
                    time.sleep(60.0)
                    self._log_start = worker_log_size(self.frames_dir/"render")
                    self.renderer = Renderer(self.frames_dir/"render", self._scene, width=self.width, height=self.height,
                                             samples=self._samples)
                    self._mesh_cache = {}
                    self.memory_restarts += 1
                    continue
                break
            except (RuntimeError, OSError, ConnectionError) as exc:
                if str(exc).startswith("film: the render worker met GPU memory"):
                    raise
                # a worker that gives no valid readback on a shared GPU (both films' workers did at once when a third
                # one started: "No valid readback after 8 render-only attempts") is replaced and the frame asked again
                print(f"film: render failed ({str(exc)[:160]}); restarting the render worker", flush=True)
                try:
                    self.renderer.close()
                except Exception:
                    pass
                attempt += 1
                if attempt == 5:
                    break
                self._log_start = worker_log_size(self.frames_dir/"render")
                self.renderer = Renderer(self.frames_dir/"render", self._scene, width=self.width, height=self.height,
                                         samples=self._samples)
                self._mesh_cache = {}                     # a new worker has none of the meshes: send them all
                self.restarts += 1
        self.render_s += time.perf_counter() - t0
        if imgs is None:
            # Repeating the previous image would label an unobserved state with the
            # previous time and conceal motion. Preserve the actual frames for
            # recovery and fail explicitly; never publish a stale image as evidence.
            raise RuntimeError(f"film: no frame at tick {w.tick} after five render workers; "
                               f"{self.written} actual frames retained in {self.frames_dir}")
        speed = f"{self.fps*self.dt:.2g}x"
        clock = self.env.material_clock()
        lines = [f"{self.title}  |  {self.assistance}",
                 f"simulated time {w.sim_time:7.2f} s (material clock {clock/60:6.2f} min)  |  playback {speed}  |  {self.note}",
                 self.eval_line]
        canvas = compose([imgs[c] for c in cams], [self.labels.get(c, c.replace("_", " ")) for c in cams], lines, self.width)
        canvas.save(self.frames_dir/f"frame_{self.written:05d}.png")
        self.times.append(w.sim_time)
        self.written += 1

    def close(self, stills=()):
        """Encode the film, a contact sheet and three review frames; returns (mp4 path, frames)."""
        if self.closed:
            return None
        self.closed = True
        if self in self.world.paste_recorders:
            self.world.paste_recorders.remove(self)
        try:
            self.renderer.close()
        except Exception:
            pass
        if self.failure:
            raise RuntimeError(f"{self.failure}; real frames retained in {self.frames_dir}")
        if not self.written:
            return None, 0
        from PIL import Image
        mp4 = self.out.with_suffix(".mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i", str(self.frames_dir/"frame_%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(mp4)],
                       check=True)
        n = self.written
        picks = np.linspace(0, n - 1, 6).astype(int)
        tiles = [Image.open(self.frames_dir/f"frame_{p:05d}.png") for p in picks]
        w0, h0 = tiles[0].size
        sheet = Image.new("RGB", (3*(w0//2), 2*(h0//2)))
        for i, im in enumerate(tiles):
            sheet.paste(im.resize((w0//2, h0//2)), ((i % 3)*(w0//2), (i//3)*(h0//2)))
        sheet.save(self.out.with_name(self.out.stem + "_sheet.jpg"), quality=88)
        review = Image.new("RGB", (w0, 3*h0))
        for i, pk in enumerate((int(0.1*(n - 1)), int(0.5*(n - 1)), int(0.9*(n - 1)))):
            review.paste(Image.open(self.frames_dir/f"frame_{pk:05d}.png"), (0, i*h0))
        review.save(self.out.with_name(self.out.stem + "_review.jpg"), quality=85)
        for k in stills:
            if 0 <= k < n:
                shutil.copy2(self.frames_dir/f"frame_{k:05d}.png", self.out.with_name(f"{self.out.stem}_f{k:05d}.png"))
        info = dict(film=mp4.name, title=self.title, assistance=self.assistance, views=self.cameras, frames=n,
                    render_worker_restarts=self.restarts, frames_held=self.frames_held,
                    workers_replaced_for_gpu_memory=self.memory_restarts,
                    frames_per_second=self.fps, states_per_simulated_s=round(1/self.dt, 3), playback=f"{self.fps*self.dt:.2g}x",
                    simulated_s=[round(self.times[0], 3), round(self.times[-1], 3)], seconds=round(n/self.fps, 2), size=[w0, h0],
                    render_s=round(self.render_s, 1), continuous="one run, one Film: no splice",
                    frame_times_s=[round(t, 6) for t in self.times],
                    sheet=self.out.stem + "_sheet.jpg", review_frames=self.out.stem + "_review.jpg")
        self.out.with_name(self.out.stem + "_film.json").write_text(json.dumps(info, indent=1) + "\n")
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        return mp4, n
