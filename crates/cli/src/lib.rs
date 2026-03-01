pub mod args;

use kbolt_core::engine::Engine;
use kbolt_core::Result;
use kbolt_types::ActiveSpaceSource;

pub struct CliAdapter {
    pub engine: Engine,
}

impl CliAdapter {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }

    pub fn space_current(&self, explicit: Option<&str>) -> Result<String> {
        let active = self.engine.current_space(explicit)?;
        let output = match active {
            Some(active) => {
                let source = match active.source {
                    ActiveSpaceSource::Flag => "flag",
                    ActiveSpaceSource::EnvVar => "env",
                    ActiveSpaceSource::ConfigDefault => "default",
                };
                format!("active space: {} ({source})", active.name)
            }
            None => "active space: none".to_string(),
        };
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    use tempfile::tempdir;

    use super::CliAdapter;
    use kbolt_core::engine::Engine;

    struct EnvRestore {
        config_home: Option<OsString>,
        cache_home: Option<OsString>,
    }

    impl EnvRestore {
        fn capture() -> Self {
            Self {
                config_home: std::env::var_os("XDG_CONFIG_HOME"),
                cache_home: std::env::var_os("XDG_CACHE_HOME"),
            }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
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

    fn with_isolated_xdg_dirs<T>(run: impl FnOnce() -> T) -> T {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let lock = ENV_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().expect("lock env mutex");
        let _restore = EnvRestore::capture();

        let root = tempdir().expect("create temp root");
        let config_home = root.path().join("config-home");
        let cache_home = root.path().join("cache-home");
        std::env::set_var("XDG_CONFIG_HOME", &config_home);
        std::env::set_var("XDG_CACHE_HOME", &cache_home);

        run()
    }

    #[test]
    fn space_current_reports_none_when_no_source_is_set() {
        with_isolated_xdg_dirs(|| {
            let mut engine = Engine::new(None).expect("create engine");
            engine
                .set_default_space(None)
                .expect("clear default space for test");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_current(None).expect("run space current");
            assert_eq!(output, "active space: none");
        });
    }

    #[test]
    fn space_current_reports_default_source() {
        with_isolated_xdg_dirs(|| {
            let mut engine = Engine::new(None).expect("create engine");
            engine
                .set_default_space(Some("default"))
                .expect("set default space");
            let adapter = CliAdapter::new(engine);

            let output = adapter.space_current(None).expect("run space current");
            assert_eq!(output, "active space: default (default)");
        });
    }
}
