# Local-only Whisper NPU prototype (Windows)

**Status:** The standalone OpenVINO GenAI pipeline and the Rust-to-Python worker pass a synthetic-speech test on an Intel Core Ultra 5 135U. The Windows GNU source build passes `cargo check` and produces a debug executable, but its test executable exits during Windows loading with `0xc0000139` (entry point not found). The full Meetily GUI has **not yet been validated** end-to-end. Do not replace a running installed Meetily executable.

The current GGML Whisper model cannot run on the NPU. The optional backend uses a separate OpenVINO-format Whisper base FP16 model with `openvino-genai==2025.4.1.0`, `openvino==2025.4.1`, and `openvino-tokenizers==2025.4.1.0`. The selected model revision is `OpenVINO/whisper-base-fp16-ov@84fbe975a79a8c996fd32c036558f29e2db6670f`. OpenVINO compiles the encoder on `EXECUTION_DEVICES: NPU` on this machine.

For this local source build, tools, caches, and model reside in `.local-tools/npu-prototype` under the repository and are git-ignored. The script `scripts/whisper_npu_worker.py` is part of the source checkout. The Windows app opts in only if the Python path is set before startup. Provide **absolute paths** to all three:

```powershell
$repo = (Resolve-Path .).Path  # run from the Meetily repository root
$env:MEETILY_WHISPER_NPU_PYTHON = "$repo\.local-tools\npu-prototype\venv\Scripts\python.exe"
$env:MEETILY_WHISPER_NPU_SCRIPT = "$repo\scripts\whisper_npu_worker.py"
$env:MEETILY_WHISPER_NPU_MODEL = "$repo\.local-tools\npu-prototype\model"
```

For the fork's **Windows portable test build**, `tauri.windows.conf.json` sets a separate app identifier (`com.meetily.npu-test`). Start the portable executable only through the repo-local `.local-tools/run-npu-test.ps1` launcher after placing it under `.local-tools/portable/`. The launcher routes app data, recordings, models and the WebView2 profile under `.local-tools/npu-test-profile/`, and refuses to start without the required environment variables. Do not run the test executable directly, or install it over the normal Meetily app.

In the new Meetily build, select **Local Whisper** as the transcription provider. The current UI still requires a downloaded GGML Whisper model for model readiness and CPU fallback. Parakeet and built-in summarization do not use this Whisper NPU worker. The worker is serialized and stays alive for repeated chunks; a failure or 150-second timeout kills it and retries the chunk with whisper.cpp. It accepts 16 kHz mono float PCM and handles up to 10 minutes per call. It runs locally without uploading audio.

On this PC a synthetic 11-second sample transcribed in ~0.3–0.5 seconds on NPU versus ~1.5–2.0 seconds on CPU after model load; a repeated 44-second sample took ~1.7–2.6 seconds on NPU versus ~4.6–5.8 seconds on CPU. Initial model compilation took ~73 seconds, with later loads ~2 seconds. These numbers are not a real-meeting accuracy or latency guarantee.

The app currently uses `whisper-rs` for normal Whisper. The NPU subprocess is an **opt-in development integration**, not yet a supported installer feature. To disable it, unset `MEETILY_WHISPER_NPU_PYTHON` before launching the source build. Never copy these variables to your normal installed 0.4.1 build and expect acceleration; that binary does not contain the integration.
