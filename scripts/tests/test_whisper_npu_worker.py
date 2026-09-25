"""Offline protocol tests for the optional Whisper NPU worker (no audio/model needed)."""
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import patch

WORKER = Path(__file__).resolve().parents[1] / "whisper_npu_worker.py"


def request(samples, language="en"):
    header = json.dumps({"samples": len(samples), "language": language}).encode()
    pcm = struct.pack(f"<{len(samples)}f", *samples)
    return struct.pack("<I", len(header)) + header + pcm


def responses(body):
    results = []
    stream = io.BytesIO(body)
    while prefix := stream.read(4):
        size = struct.unpack("<I", prefix)[0]
        results.append(json.loads(stream.read(size)))
    return results


class FakePipeline:
    instances = []

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.calls = []
        self.instances.append(self)

    def generate(self, audio, **options):
        self.calls.append((audio, options))
        return " synthetic speech "


class WorkerProtocolTests(unittest.TestCase):
    def run_worker(self, data):
        fake_numpy = types.ModuleType("numpy")

        def frombuffer(audio, dtype):
            self.assertEqual(dtype, "<f4")
            samples = struct.unpack(f"<{len(audio) // 4}f", audio)
            return types.SimpleNamespace(tolist=lambda: list(samples))

        fake_numpy.frombuffer = frombuffer
        fake_genai = types.ModuleType("openvino_genai")
        fake_genai.WhisperPipeline = FakePipeline
        stdout = io.BytesIO()
        FakePipeline.instances.clear()
        spec = importlib.util.spec_from_file_location("test_npu_worker", WORKER)
        worker = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"numpy": fake_numpy, "openvino_genai": fake_genai}), \
             patch.dict(os.environ, {"MEETILY_WHISPER_NPU_MODEL": "synthetic-model"}), \
             patch.object(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(data))), \
             patch.object(sys, "stdout", types.SimpleNamespace(buffer=stdout)):
            spec.loader.exec_module(worker)
            worker.main()
        return responses(stdout.getvalue()), FakePipeline.instances[0]

    def test_multiple_requests_share_one_npu_pipeline(self):
        output, pipeline = self.run_worker(request([0.25, -0.5], "en") + request([0.1], "auto-translate"))
        self.assertEqual(output, [{"text": "synthetic speech"}, {"text": "synthetic speech"}])
        self.assertEqual((pipeline.model, pipeline.device), ("synthetic-model", "NPU"))
        self.assertEqual(pipeline.calls[0], ([0.25, -0.5], {"task": "transcribe", "max_new_tokens": 448, "language": "<|en|>"}))
        self.assertEqual(pipeline.calls[1][1]["task"], "translate")

    def test_bad_language_returns_error_without_losing_next_request(self):
        output, pipeline = self.run_worker(request([0.25], "bad") + request([0.5], "auto"))
        self.assertEqual(output, [{"error": "unsupported language setting"}, {"text": "synthetic speech"}])
        self.assertEqual(len(pipeline.calls), 1)

    def test_rejects_invalid_audio_count_before_reading_samples(self):
        header = json.dumps({"samples": 0, "language": "en"}).encode()
        with self.assertRaisesRegex(ValueError, "invalid audio sample count"):
            self.run_worker(struct.pack("<I", len(header)) + header)


if __name__ == "__main__":
    unittest.main()
