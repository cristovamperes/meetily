# Experimental Whisper transcription on Intel NPU (Windows)

This is an **opt-in local backend for Whisper transcription**, not a general GPU/NPU setting. It does not accelerate Parakeet or meeting summaries. The default remains Meetily's whisper.cpp backend, which is also used if the worker fails or times out. No audio is sent to a server by this backend.

## What it runs

On Windows, Meetily can start a persistent Python/OpenVINO GenAI subprocess for 16 kHz mono Whisper chunks. It is used by live recording, audio import, Enhance/retranscription, and the direct Whisper command. OpenVINO compiles the model for `NPU`; decoding, VAD and other work can still use the CPU. Requests are serialized; an individual request has a 150-second deadline and falls back to whisper.cpp on failure.

**Model selection caveat:** This prototype uses a separate OpenVINO **Whisper base FP16** model, *regardless of the GGML Whisper model selected in Meetily*. The GGML model is still needed for the existing UI/readiness flow and CPU fallback. Transcription accuracy may therefore differ from the selected model. The displayed confidence is Meetily's existing text-length heuristic, not a calibrated probability. Do not assume equal quality to a large/turbo GGML model.

## Local source-build setup

Requirements: Windows with a supported Intel NPU and driver, a working Python installation, and `openvino-genai==2025.4.1.0`, `openvino==2025.4.1`, `openvino-tokenizers==2025.4.1.0` and `numpy` in a dedicated environment. Download an **OpenVINO-format** Whisper model locally; a `ggml-*.bin` model cannot be used by this worker. An example model is `OpenVINO/whisper-base-fp16-ov` at revision `84fbe975a79a8c996fd32c036558f29e2db6670f`. The model directory must contain `openvino_encoder_model.xml` and the remaining files required by OpenVINO GenAI.

Set all three absolute paths **before starting Meetily** (adjust for your checkout):

```powershell
$repo = (Resolve-Path .).Path
$env:MEETILY_WHISPER_NPU_PYTHON = "$repo\.local-tools\whisper-npu\venv\Scripts\python.exe"
$env:MEETILY_WHISPER_NPU_SCRIPT = "$repo\scripts\whisper_npu_worker.py"
$env:MEETILY_WHISPER_NPU_MODEL = "$repo\.local-tools\whisper-npu\model"
# Launch the Meetily source build from this PowerShell session.
```

`.local-tools/` is git-ignored. The Python environment and OpenVINO model are **not bundled or automatically downloaded** by this change. Do not set these variables for an unmodified Meetily executable: it cannot use the worker. To disable the experimental backend, unset `MEETILY_WHISPER_NPU_PYTHON` and relaunch. If any required file is missing, the worker errors, or no NPU is available, Meetily logs a warning and uses whisper.cpp for that chunk. Keep a GGML model installed for this fallback.

## Testing

Run the dependency-free worker protocol tests with `python -m unittest discover -s scripts/tests -p 'test_whisper_npu_worker.py'`. They use only synthetic in-memory audio and a fake OpenVINO pipeline. A Rust NPU smoke test is marked `#[ignore]` because it needs actual hardware, a locally configured OpenVINO installation, and a synthetic 16 kHz PCM WAV (`MEETILY_NPU_TEST_WAV`). Also verify live/Enhance and CPU fallback on synthetic recordings before adopting it for important meetings.
