"""Optional Gemini video API for the evidence auditor; no simulator interaction."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys
import time
import warnings


def review(video, question, *, model="gemini-3.8-flash", timeout_sec=600, client=None):
    video = Path(video).resolve(strict=True)
    if not video.is_file() or not question.strip() or timeout_sec <= 0:
        raise ValueError("Provide a video file, a question and a positive timeout.")
    owned = client is None
    if owned:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use optional video review.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise RuntimeError("Install the video-audit extra: pip install -e '.[video-audit]'") from error
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=timeout_sec * 1000))
    uploaded = None
    try:
        uploaded = client.files.upload(file=str(video))
        deadline = time.monotonic() + timeout_sec
        while True:
            state = getattr(uploaded.state, "name", uploaded.state)
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise RuntimeError("Gemini could not process the uploaded video.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for Gemini video processing.")
            time.sleep(min(2, remaining))
            uploaded = client.files.get(name=uploaded.name)
        response = client.interactions.create(
            model=model,
            input=[{"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type},
                   {"type": "text", "text": question}],
        )
        text = response.output_text
        if not text or not text.strip():
            raise RuntimeError("Gemini returned no video assessment.")
        return {"video": str(video), "model": model, "question": question, "assessment": text}
    finally:
        try:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception as error:
                    warnings.warn(f"Could not remove uploaded Gemini file {uploaded.name}: {type(error).__name__}")
        finally:
            if owned:
                client.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True, help="New JSON assessment file")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        result = review(args.video, args.question, model=args.model, timeout_sec=args.timeout_sec)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except Exception as error:
        message = str(error)
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = os.environ.get(name)
            if key:
                message = message.replace(key, "[redacted]")
        message = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[redacted]", message)
        print(f"Video review failed ({type(error).__name__}): {message}", file=sys.stderr)
        return 1
    print(f"Video assessment: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
