//! Isolated data path for the Windows NPU test build. The normal app is unchanged.
use std::path::PathBuf;
use tauri::{path::PathResolver, Runtime};

pub fn app_data_dir<R: Runtime>(_resolver: &PathResolver<R>) -> tauri::Result<PathBuf> {
    #[cfg(windows)]
    {
        // The fork's Windows app is test-only: never silently use the installed app's profile
        // or create test models/recordings outside the repository. The launcher supplies this.
        let value = std::env::var_os("MEETILY_NPU_TEST_DIR").ok_or(tauri::Error::UnknownPath)?;
        let path = PathBuf::from(value);
        if !path.is_absolute() || !path.is_dir() {
            return Err(tauri::Error::UnknownPath);
        }
        return Ok(path);
    }
    #[cfg(not(windows))]
    _resolver.app_data_dir()
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
