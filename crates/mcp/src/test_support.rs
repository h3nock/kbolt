use std::ffi::OsString;
use std::sync::{Mutex, OnceLock};

use tempfile::tempdir;

struct EnvRestore {
    home: Option<OsString>,
    config_home: Option<OsString>,
    cache_home: Option<OsString>,
}

impl EnvRestore {
    fn capture() -> Self {
        Self {
            home: std::env::var_os("HOME"),
            config_home: std::env::var_os("XDG_CONFIG_HOME"),
            cache_home: std::env::var_os("XDG_CACHE_HOME"),
        }
    }
}

impl Drop for EnvRestore {
    fn drop(&mut self) {
        match &self.home {
            Some(path) => std::env::set_var("HOME", path),
            None => std::env::remove_var("HOME"),
        }
        match &self.config_home {
            Some(path) => std::env::set_var("XDG_CONFIG_HOME", path),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
        match &self.cache_home {
            Some(path) => std::env::set_var("XDG_CACHE_HOME", path),
            None => std::env::remove_var("XDG_CACHE_HOME"),
        }
    }
}

pub(crate) fn with_isolated_xdg_dirs<T>(run: impl FnOnce() -> T) -> T {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let _restore = EnvRestore::capture();

    let root = tempdir().expect("create temp root");
    std::env::set_var("HOME", root.path());
    let config_home = root.path().join("config-home");
    let cache_home = root.path().join("cache-home");
    std::env::set_var("XDG_CONFIG_HOME", &config_home);
    std::env::set_var("XDG_CACHE_HOME", &cache_home);

    run()
}
