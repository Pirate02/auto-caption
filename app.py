"""A small local video captioner. Video, audio, transcripts, and exports stay on this computer."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
JOBS_DIR = DATA / "jobs"
MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"
MAX_UPLOAD = int(os.environ.get("CAPTION_MAX_UPLOAD_GB", "20")) * 1024**3
CHUNK_SECONDS = 24
FONT_DIR = ROOT / "assets" / "fonts"
FONT_FILES = {"bounce": "Poppins-ExtraBold.ttf", "hype": "Anton-Regular.ttf", "clean": "Poppins-ExtraBold.ttf"}
STYLES = {
    "bounce": {"label": "Bounce", "active": "00F0FF", "pop": 102, "upper": True},
    "hype": {"label": "Hype", "active": "3C3CFF", "pop": 102, "upper": True},
    "clean": {"label": "Clean", "active": "6BD7FF", "pop": 102, "upper": False},
}

DATA.mkdir(exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Short Captioner", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
_lock = threading.RLock()
_model_lock = threading.Lock()
_model_state = {"downloading": False, "percent": 0, "error": ""}
_recognizer = None
_jobs: dict[str, dict] = {}


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe() -> str:
    return shutil.which("ffprobe") or "ffprobe"


def _model_ready(path: Path) -> bool:
    return (path / "tokens.txt").is_file() and all((path / f"{part}.int8.onnx").is_file()
                                                    for part in ("encoder", "decoder", "joiner"))


def _model_path() -> Path:
    configured = os.environ.get("CAPTION_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    shared = ROOT.parent / "peakcut" / "data" / "models" / MODEL_NAME
    return shared if _model_ready(shared) else DATA / "models" / MODEL_NAME


def _model_source() -> str:
    path = _model_path()
    return "shared with PeakCut" if "peakcut" in path.parts else "stored in this project"


def _save_job(job: dict) -> None:
    target = JOBS_DIR / job["id"] / "job.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in ("id", "name", "status", "stage", "progress", "error", "duration",
                                    "created", "style", "framing", "video_url", "srt_url")}


def _load_jobs() -> None:
    for file in JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(file.read_text(encoding="utf-8"))
            if job.get("status") in ("queued", "running"):
                job.update(status="error", stage="Stopped", error="The app closed while this video was processing. Start it again to try once more.")
                _save_job(job)
            _jobs[job["id"]] = job
        except (OSError, ValueError, KeyError):
            continue


_load_jobs()


def _model_download() -> None:
    with _model_lock:
        if _model_state["downloading"] or _model_ready(_model_path()):
            return
        _model_state.update(downloading=True, percent=0, error="")
        target = _model_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="caption-model-", dir=target.parent))
        archive = tmp / "model.tar.bz2"
        unpack = tmp / "unpacked"
        try:
            def progress(blocks, block_size, total):
                if total > 0:
                    _model_state["percent"] = min(99, int(blocks * block_size * 100 / total))

            urlretrieve(MODEL_URL, archive, progress)
            unpack.mkdir()
            with tarfile.open(archive, "r:bz2") as tar:
                for item in tar.getmembers():
                    parts = Path(item.name).parts
                    if not parts or Path(item.name).is_absolute() or ".." in parts or not item.isfile():
                        continue
                    out = unpack.joinpath(*parts)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    src = tar.extractfile(item)
                    if src:
                        with src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
            unpacked = unpack / MODEL_NAME
            if not _model_ready(unpacked):
                raise RuntimeError("The model download did not contain all required files.")
            if target.exists():
                shutil.rmtree(target)
            unpacked.rename(target)
            _model_state["percent"] = 100
        except Exception as exc:  # noqa: BLE001
            _model_state["error"] = str(exc)[-300:]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            _model_state["downloading"] = False


def _get_recognizer():
    global _recognizer
    if _recognizer is not None:
        return _recognizer
    import sherpa_onnx

    model = _model_path()
    _recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model / "encoder.int8.onnx"), decoder=str(model / "decoder.int8.onnx"),
        joiner=str(model / "joiner.int8.onnx"), tokens=str(model / "tokens.txt"),
        num_threads=max(2, min(6, (os.cpu_count() or 4) - 2)), sample_rate=16000, feature_dim=80,
        decoding_method="greedy_search", model_type="nemo_transducer", provider="cpu")
    return _recognizer


def _words(tokens, stamps, offset: float, duration: float) -> list[dict]:
    result = []
    current = None
    gap = False
    for token, stamp in zip(tokens, stamps):
        new_word = token.startswith((" ", "▁")) or gap
        piece = token.replace("▁", " ").strip()
        if not piece:
            gap = True
            continue
        gap = False
        is_punctuation = all(not char.isalnum() for char in piece)
        if current is None or (new_word and not is_punctuation):
            if current:
                result.append(current)
            current = {"w": piece, "s": round(offset + float(stamp), 2),
                       "e": round(offset + float(stamp) + 0.12, 2)}
        else:
            current["w"] += piece
            current["e"] = round(offset + float(stamp) + 0.12, 2)
    if current:
        current["e"] = min(current["e"], round(offset + duration, 2))
        result.append(current)
    return result


def _transcribe(path: Path, duration: float, job: dict) -> list[dict]:
    rec = _get_recognizer()
    words: list[dict] = []
    core_start = 0.0
    while core_start < duration:
        core_end = min(duration, core_start + CHUNK_SECONDS)
        read_start = max(0.0, core_start - (0.6 if core_start else 0.0))
        read_end = min(duration, core_end + 0.6)
        span = read_end - read_start
        cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-ss", f"{read_start:.2f}", "-i", str(path),
               "-t", f"{span:.2f}", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
               "-f", "s16le", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True, timeout=max(120, span * 6))
        if proc.returncode:
            detail = proc.stderr.decode(errors="replace")[-350:]
            raise RuntimeError(detail or "Could not read audio from this video.")
        audio = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
        if len(audio) > 1600 and float(np.sqrt(np.mean(audio * audio))) >= 0.003:
            stream = rec.create_stream()
            stream.accept_waveform(16000, audio)
            rec.decode_stream(stream)
            got = _words(list(stream.result.tokens), [float(x) for x in stream.result.timestamps], read_start, span)
            words.extend(w for w in got if core_start <= (w["s"] + w["e"]) / 2 < core_end)
        core_start = core_end
        job.update(stage="Transcribing speech", progress=0.12 + 0.72 * core_start / max(duration, 1))
        _save_job(job)
    for i, word in enumerate(words[:-1]):
        word["e"] = max(word["s"], min(word["e"], words[i + 1]["s"]))
    return words


def _ass_time(seconds: float) -> str:
    cs = max(0, int(round(seconds * 100)))
    hours, cs = divmod(cs, 360000)
    minutes, cs = divmod(cs, 6000)
    seconds_i, cs = divmod(cs, 100)
    return f"{hours}:{minutes:02d}:{seconds_i:02d}.{cs:02d}"


def _ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def _groups(words: list[dict], per: int) -> list[list[dict]]:
    out, current = [], []
    for word in words:
        if current and (len(current) >= per or word["s"] - current[-1]["e"] > 0.42
                        or re.search(r"[.!?]$", current[-1]["w"])):
            out.append(current)
            current = []
        current.append(word)
    if current:
        out.append(current)
    return out


def _subtitle_font_measure(font_path: Path):
    """Measure text at the effective size libass uses for ASS font sizes."""
    import struct
    from PIL import ImageFont

    data = font_path.read_bytes()
    tables = {}
    for i in range(struct.unpack(">H", data[4:6])[0]):
        tag, _, offset, _ = struct.unpack(">4sIII", data[12 + 16 * i:28 + 16 * i])
        tables[tag] = offset
    units_per_em = struct.unpack(">H", data[tables[b"head"] + 18:tables[b"head"] + 20])[0]
    os2 = tables[b"OS/2"]
    ascent, descent = struct.unpack(">HH", data[os2 + 74:os2 + 78])
    font = ImageFont.truetype(str(font_path), 200)
    return lambda text, size: font.getlength(text) * size * units_per_em / max(1, ascent + descent) / 200


def _build_subtitles(words: list[dict], settings: dict, width: int, height: int) -> tuple[str, str]:
    style = STYLES[settings["style"]]
    font_path = FONT_DIR / FONT_FILES[settings["style"]]
    # Choose one size for the whole video. Fit every phrase against that same
    # size so a long phrase does not suddenly look much smaller than the rest.
    font_size = int(min(height * settings["size"], width * 0.12))
    groups = _groups(words, settings["per"])
    measure = _subtitle_font_measure(font_path)
    widest_phrase = max(
        (sum(measure(w["w"].upper() if settings["upper"] else w["w"], font_size) for w in group)
         + measure(" ", font_size) * max(0, len(group) - 1) for group in groups),
        default=0,
    )
    if widest_phrase > width * 0.88:
        font_size = max(24, int(font_size * (width * 0.88 / widest_phrase)))
    outline = max(3, int(height * 0.004))
    align, y_ratio = {"top": (8, 0.14), "middle": (5, 0.50), "bottom": (2, 0.79)}[settings["position"]]
    y = int(height * y_ratio)
    style_head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{'Anton' if settings['style'] == 'hype' else 'Poppins ExtraBold'},{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,0,0,0,0,100,100,0,0,1,{outline},0,{align},0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    srt = []
    for index, group in enumerate(groups):
        texts = [w["w"].upper() if settings["upper"] else w["w"] for w in group]
        word_widths = [measure(text, font_size) for text in texts]
        space_width = measure(" ", font_size)
        x = (width - sum(word_widths) - space_width * max(0, len(group) - 1)) / 2
        g_start = max(0.0, group[0]["s"] - 0.02)
        next_start = groups[index + 1][0]["s"] if index + 1 < len(groups) else float("inf")
        # End a phrase as the next one begins. Otherwise phrases overlap on screen and look like doubled captions.
        g_end = min(group[-1]["e"] + 0.16, next_start)
        if g_end <= g_start:
            continue
        srt_text = " ".join(texts)
        srt.append(f"{index + 1}\n{_srt_time(g_start)} --> {_srt_time(g_end)}\n{srt_text}\n")
        for word, text, word_width in zip(group, texts, word_widths):
            word_pos = f"\\an{align}\\pos({int(x + word_width / 2)},{y})\\fs{font_size}"
            start = max(g_start, word["s"])
            end = min(g_end, word["e"])
            safe_text = _ass_text(text)
            if not settings["highlight"] or end <= start:
                events.append(f"Dialogue: 1,{_ass_time(g_start)},{_ass_time(g_end)},Cap,,0,0,0,,{{{word_pos}\\c&HFFFFFF&}}{safe_text}")
            else:
                if start - g_start >= 0.02:
                    events.append(f"Dialogue: 1,{_ass_time(g_start)},{_ass_time(start)},Cap,,0,0,0,,{{{word_pos}\\c&HFFFFFF&}}{safe_text}")
                transform = (f"\\fscx100\\fscy100\\t(0,55,\\fscx{style['pop']}\\fscy{style['pop']})"
                             f"\\t(55,120,\\fscx100\\fscy100)") if settings["animate"] else ""
                events.append(f"Dialogue: 2,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{{{word_pos}\\c&H{style['active']}&{transform}}}{safe_text}")
                if g_end - end >= 0.02:
                    events.append(f"Dialogue: 1,{_ass_time(end)},{_ass_time(g_end)},Cap,,0,0,0,,{{{word_pos}\\c&HFFFFFF&}}{safe_text}")
            x += word_width + space_width
    return style_head + "\n".join(events) + "\n", "\n".join(srt)


def _srt_time(seconds: float) -> str:
    ms = max(0, int(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    seconds_i, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds_i:02},{ms:03}"


def _probe(path: Path) -> tuple[float, int, int]:
    cmd = [_ffprobe(), "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
           "-of", "json", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-400:] or "FFprobe could not read this video.")
    meta = json.loads(proc.stdout)
    video = next((s for s in meta.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("This file does not contain a video track.")
    if not any(s.get("codec_type") == "audio" for s in meta.get("streams", [])):
        raise RuntimeError("This video has no audio track to caption.")
    duration = float(meta.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        raise RuntimeError("Could not read the video duration.")
    return duration, int(video["width"]), int(video["height"])


def _render(job: dict, width: int, height: int) -> None:
    folder = JOBS_DIR / job["id"]
    ass_path = folder / "captions.ass"
    settings = job["settings"]
    out_width, out_height = (1080, 1920) if settings["framing"] == "vertical" else (width, height)
    ass, srt = _build_subtitles(job["words"], settings, out_width, out_height)
    ass_path.write_text(ass, encoding="utf-8")
    (folder / "captions.srt").write_text(srt, encoding="utf-8")
    inp, out = folder / "input" / job["stored_name"], folder / "captioned.mp4"
    vf = f"ass=filename=captions.ass:fontsdir={FONT_DIR.as_posix()}"
    cmd = [_ffmpeg(), "-y", "-nostdin", "-v", "error", "-i", str(inp)]
    if settings["framing"] == "vertical":
        vf_graph = ("[0:v]split=2[bg][fg];"
                    "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=18:4[bg];"
                    "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
                    "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base];"
                    f"[base]ass=filename=captions.ass:fontsdir={FONT_DIR.as_posix()}[v]")
        cmd += ["-filter_complex", vf_graph, "-map", "[v]"]
    else:
        cmd += ["-vf", vf, "-map", "0:v:0"]
    cmd += ["-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=max(600, job["duration"] * 8))
    if proc.returncode:
        raise RuntimeError(proc.stderr[-600:] or "FFmpeg could not render the captioned video.")


def _process(job_id: str) -> None:
    job = _jobs[job_id]
    try:
        job.update(status="running", stage="Checking video", progress=0.05)
        _save_job(job)
        folder = JOBS_DIR / job_id
        inp = folder / "input" / job["stored_name"]
        duration, width, height = _probe(inp)
        job["duration"] = duration
        job.update(stage="Loading local speech model", progress=0.1)
        _save_job(job)
        if not _model_ready(_model_path()):
            _model_download()
            if not _model_ready(_model_path()):
                raise RuntimeError(_model_state.get("error") or "The local speech model could not be downloaded.")
        job["words"] = _transcribe(inp, duration, job)
        if not job["words"]:
            raise RuntimeError("No speech was detected. Try a video with clearer spoken audio.")
        job.update(stage="Burning captions into video", progress=0.87)
        _save_job(job)
        _render(job, width, height)
        job.update(status="done", stage="Finished", progress=1.0, words=None,
                   video_url=f"/api/jobs/{job_id}/video", srt_url=f"/api/jobs/{job_id}/srt", error="")
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", stage="Needs attention", error=str(exc)[-600:], words=None)
    finally:
        job["finished"] = time.time()
        _save_job(job)


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    ffmpeg_ok = False
    try:
        p = subprocess.run([_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=12)
        ffmpeg_ok = p.returncode == 0 and bool(re.search(r"\bass\s+V->V", p.stdout))
    except Exception:  # noqa: BLE001
        pass
    return {"model_ready": _model_ready(_model_path()), "model_downloading": _model_state["downloading"],
            "model_percent": _model_state["percent"], "model_error": _model_state["error"],
            "model_source": _model_source(), "ffmpeg_ass": ffmpeg_ok}


@app.post("/api/model")
def download_model():
    if not _model_ready(_model_path()) and not _model_state["downloading"]:
        threading.Thread(target=_model_download, daemon=True).start()
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs():
    with _lock:
        ordered = sorted(_jobs.values(), key=lambda x: x.get("created", 0), reverse=True)
        return [_public(j) for j in ordered[:30]]


@app.post("/api/jobs")
async def create_job(video: UploadFile = File(...), style: str = Form("bounce"), position: str = Form("bottom"),
                     per: int = Form(3), size: float = Form(0.065), upper: bool = Form(True),
                     highlight: bool = Form(True), animate: bool = Form(True), framing: str = Form("original")):
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}:
        raise HTTPException(400, "Choose a video file: MP4, MOV, M4V, MKV, WebM, or AVI.")
    if style not in STYLES or position not in {"top", "middle", "bottom"} or framing not in {"original", "vertical"}:
        raise HTTPException(400, "One of the caption options is not supported.")
    if not 1 <= per <= 4 or not 0.035 <= size <= 0.095:
        raise HTTPException(400, "Caption size or words-per-line is outside the supported range.")
    ffmpeg_info = status()
    if not ffmpeg_info["ffmpeg_ass"]:
        raise HTTPException(503, "FFmpeg with ASS subtitle support is required. On Mac, install ffmpeg-full and start the app with ./run.sh.")
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS_DIR / job_id
    input_dir = folder / "input"
    input_dir.mkdir(parents=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(video.filename or "video.mp4").name).strip("._") or "video.mp4"
    stored_name = f"source{suffix}"
    target = input_dir / stored_name
    total = 0
    try:
        with target.open("wb") as out:
            while chunk := await video.read(4 * 1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD:
                    raise HTTPException(413, f"Video exceeds the {MAX_UPLOAD // 1024**3} GB upload limit.")
                out.write(chunk)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    finally:
        await video.close()
    if total < 2048:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "That video file is too small to process.")
    settings = {"style": style, "position": position, "per": per, "size": size,
                "upper": upper, "highlight": highlight, "animate": animate, "framing": framing}
    job = {"id": job_id, "name": safe_name, "stored_name": stored_name, "status": "queued", "stage": "Waiting to start",
           "progress": 0.02, "error": "", "created": time.time(), "settings": settings,
           "style": STYLES[style]["label"], "framing": framing, "duration": 0}
    with _lock:
        _jobs[job_id] = job
        _save_job(job)
    threading.Thread(target=_process, args=(job_id,), daemon=True).start()
    return JSONResponse(_public(job))


@app.get("/api/jobs/{job_id}/video")
def download_video(job_id: str):
    job = _jobs.get(job_id)
    path = JOBS_DIR / job_id / "captioned.mp4"
    if not job or job.get("status") != "done" or not path.is_file():
        raise HTTPException(404, "Captioned video not found.")
    return FileResponse(path, media_type="video/mp4", filename=f"{Path(job['name']).stem}-captioned.mp4")


@app.get("/api/jobs/{job_id}/srt")
def download_srt(job_id: str):
    job = _jobs.get(job_id)
    path = JOBS_DIR / job_id / "captions.srt"
    if not job or job.get("status") != "done" or not path.is_file():
        raise HTTPException(404, "Caption file not found.")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{Path(job['name']).stem}-captions.srt")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CAPTION_PORT", "8766")))
