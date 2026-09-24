//! Opt-in, local-only OpenVINO Whisper worker. A separate process keeps the
//! OpenVINO model compiled between meeting chunks and isolates its C++ runtime
//! from Meetily's existing ONNX Runtime and whisper.cpp runtimes.
use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use serde_json::json;
use std::{path::PathBuf, process::Stdio, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

const MAX_SAMPLES: usize = 16_000 * 600;
const MAX_RESPONSE: usize = 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(150);
static WORKER: Lazy<Mutex<Option<Worker>>> = Lazy::new(|| Mutex::new(None));

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

fn configured_paths() -> Option<Result<(PathBuf, PathBuf, PathBuf)>> {
    let python = std::env::var_os("MEETILY_WHISPER_NPU_PYTHON")?;
    Some((|| {
        let python = PathBuf::from(python);
        let script = PathBuf::from(
            std::env::var_os("MEETILY_WHISPER_NPU_SCRIPT")
                .ok_or_else(|| anyhow!("MEETILY_WHISPER_NPU_SCRIPT is not set"))?,
        );
        let model = PathBuf::from(
            std::env::var_os("MEETILY_WHISPER_NPU_MODEL")
                .ok_or_else(|| anyhow!("MEETILY_WHISPER_NPU_MODEL is not set"))?,
        );
        if !python.is_absolute()
            || !python.is_file()
            || !script.is_absolute()
            || !script.is_file()
            || !model.is_absolute()
            || !model.join("openvino_encoder_model.xml").is_file()
        {
            return Err(anyhow!(
                "NPU Python, worker script, and model must be existing absolute paths"
            ));
        }
        Ok((python, script, model))
    })())
}

impl Worker {
    fn spawn(python: &PathBuf, script: &PathBuf, model: &PathBuf) -> Result<Self> {
        let mut child = Command::new(python)
            .arg("-u")
            .arg(script)
            .env("MEETILY_WHISPER_NPU_MODEL", model)
            .env("PYTHONNOUSERSITE", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("HF_HUB_DISABLE_TELEMETRY", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .context("Failed to start local Whisper NPU worker")?;
        let stdin = child.stdin.take().context("NPU worker has no stdin")?;
        let stdout = child.stdout.take().context("NPU worker has no stdout")?;
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    async fn transcribe(&mut self, samples: &[f32], language: Option<&str>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        if samples.len() > MAX_SAMPLES {
            return Err(anyhow!("NPU audio chunk exceeds 10 minutes"));
        }
        let header = serde_json::to_vec(&json!({
            "samples": samples.len(),
            "language": language.unwrap_or("auto"),
        }))?;
        let header_size = u32::try_from(header.len())?;
        let mut pcm = Vec::with_capacity(samples.len() * 4);
        for sample in samples {
            pcm.extend_from_slice(&sample.to_le_bytes());
        }
        self.stdin.write_all(&header_size.to_le_bytes()).await?;
        self.stdin.write_all(&header).await?;
        self.stdin.write_all(&pcm).await?;
        self.stdin.flush().await?;

        let mut length = [0u8; 4];
        self.stdout.read_exact(&mut length).await?;
        let size = u32::from_le_bytes(length) as usize;
        if size == 0 || size > MAX_RESPONSE {
            return Err(anyhow!("Invalid NPU worker response length"));
        }
        let mut response = vec![0; size];
        self.stdout.read_exact(&mut response).await?;
        let response: serde_json::Value = serde_json::from_slice(&response)?;
        if let Some(error) = response.get("error").and_then(|value| value.as_str()) {
            return Err(anyhow!("NPU transcription failed: {error}"));
        }
        Ok(response
            .get("text")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("Missing NPU transcription text"))?
            .to_string())
    }
}

/// None means disabled. An error means the caller should use whisper.cpp instead.
pub async fn try_transcribe(samples: &[f32], language: Option<&str>) -> Option<Result<String>> {
    let paths = match configured_paths()? {
        Ok(paths) => paths,
        Err(error) => return Some(Err(error)),
    };
    let mut guard = WORKER.lock().await;
    let launched = guard.is_none();
    if launched {
        match Worker::spawn(&paths.0, &paths.1, &paths.2) {
            Ok(worker) => *guard = Some(worker),
            Err(error) => return Some(Err(error)),
        }
    }
    let result = timeout(
        REQUEST_TIMEOUT,
        guard
            .as_mut()
            .expect("NPU worker was initialized")
            .transcribe(samples, language),
    )
    .await;
    match result {
        Ok(Ok(text)) => {
            if launched {
                log::info!("Whisper OpenVINO NPU transcription active");
            }
            Some(Ok(text))
        }
        Ok(Err(error)) => {
            if let Some(mut worker) = guard.take() {
                let _ = worker.child.kill().await;
            }
            Some(Err(error))
        }
        Err(_) => {
            if let Some(mut worker) = guard.take() {
                let _ = worker.child.kill().await;
            }
            Some(Err(anyhow!("NPU transcription timed out")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::try_transcribe;

    #[tokio::test]
    #[ignore = "requires local OpenVINO installation, NPU and synthetic test WAV"]
    async fn transcribes_twice_on_the_same_worker() {
        let wav = std::fs::read(std::env::var("MEETILY_NPU_TEST_WAV").unwrap()).unwrap();
        let data = wav.windows(4).position(|bytes| bytes == b"data").unwrap() + 8;
        let audio: Vec<f32> = wav[data..]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0)
            .collect();
        let first = try_transcribe(&audio, Some("en")).await.unwrap().unwrap();
        let second = try_transcribe(&audio, Some("en")).await.unwrap().unwrap();
        assert!(
            first.contains("Friday"),
            "Unexpected synthetic transcription"
        );
        assert_eq!(first, second);
    }
}
