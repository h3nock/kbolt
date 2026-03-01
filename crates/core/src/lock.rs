use std::fs::{self, File, OpenOptions};
use std::path::Path;

use fs2::FileExt;
use kbolt_types::KboltError;

use crate::Result;

const LOCK_FILENAME: &str = "kbolt.lock";
const LOCK_UNAVAILABLE_MESSAGE: &str = "Another kbolt process is active. Try again shortly.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    Shared,
    Exclusive,
}

pub struct OperationLock {
    file: File,
}

impl OperationLock {
    pub fn acquire(cache_dir: &Path, mode: LockMode) -> Result<Self> {
        fs::create_dir_all(cache_dir)?;
        let lock_path = cache_dir.join(LOCK_FILENAME);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;

        let lock_result = match mode {
            LockMode::Shared => FileExt::try_lock_shared(&file),
            LockMode::Exclusive => FileExt::try_lock_exclusive(&file),
        };

        if let Err(err) = lock_result {
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Err(KboltError::Internal(LOCK_UNAVAILABLE_MESSAGE.to_string()).into());
            }
            return Err(err.into());
        }

        Ok(Self { file })
    }
}

impl Drop for OperationLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}
