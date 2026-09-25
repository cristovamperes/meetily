//! Optional isolated data directory; absent in the normal installed app.
use std::path::PathBuf;
use tauri::{path::PathResolver, Runtime};

pub fn app_data_dir<R: Runtime>(resolver: &PathResolver<R>) -> tauri::Result<PathBuf> {
    if let Some(value) = std::env::var_os("MEETILY_NPU_TEST_DIR") {
        let path = PathBuf::from(value);
        if !path.is_absolute() || !path.is_dir() {
            return Err(tauri::Error::UnknownPath);
        }
        return Ok(path);
    }
    resolver.app_data_dir()
}

pub fn store_path(filename: &str) -> PathBuf {
    portable_dir().map_or_else(|| PathBuf::from(filename), |dir| dir.join(filename))
}

pub fn portable_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        let path = PathBuf::from(std::env::var_os("MEETILY_NPU_TEST_DIR")?);
        if path.is_absolute() && path.is_dir() { Some(path) } else { None }
    }
    #[cfg(not(windows))]
    None
}
