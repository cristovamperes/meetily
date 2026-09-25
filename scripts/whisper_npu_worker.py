"""Local OpenVINO GenAI Whisper worker. Protocol: u32le JSON-header length,
JSON {samples: int, language: str}, float32le PCM at 16 kHz; response is
u32le JSON length, JSON {text: str} or {error: str}. Stdout is protocol only.
"""
import json
import os
import struct
import sys

import numpy as np
import openvino_genai as genai

MAX_SAMPLES = 16000 * 600  # 10 minutes per request


def read_exact(size):
    data = bytearray(size)
    view = memoryview(data)
    while view:
        count = sys.stdin.buffer.readinto(view)
        if not count:
            raise EOFError("worker input closed")
        view = view[count:]
    return data


def respond(value):
    body = json.dumps(value, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(body)) + body)
    sys.stdout.buffer.flush()


def main():
    model = os.environ["MEETILY_WHISPER_NPU_MODEL"]
    pipeline = genai.WhisperPipeline(model, "NPU")
    while True:
        prefix = sys.stdin.buffer.read(4)
        if not prefix:
            return
        if len(prefix) != 4:
            raise EOFError("incomplete request length")
        size = struct.unpack("<I", prefix)[0]
        if not 0 < size <= 1024:
            raise ValueError("invalid request header length")
        header = json.loads(read_exact(size))
        count = header["samples"]
        if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= MAX_SAMPLES:
            raise ValueError("invalid audio sample count")
        audio = np.frombuffer(read_exact(count * 4), dtype="<f4")
        language = header.get("language", "auto")
        options = {"task": "transcribe", "max_new_tokens": 448}
        if language == "auto-translate":
            options["task"] = "translate"
        elif isinstance(language, str) and len(language) == 2 and language.isalpha():
            options["language"] = f"<|{language.lower()}|>"
        elif language not in (None, "auto"):
            respond({"error": "unsupported language setting"})
            continue
        try:
            result = pipeline.generate(audio.tolist(), **options)
            respond({"text": str(result).strip()})
        except Exception as exc:
            # Do not print audio or transcript to logs.
            respond({"error": str(exc)[:512]})


if __name__ == "__main__":
    main()
