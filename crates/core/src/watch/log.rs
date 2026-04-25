use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::Result;

use super::{WATCH_LOG_MAX_BYTES, WATCH_LOG_ROTATIONS};

const LOG_DIR: &str = "logs";
const WATCH_LOG_FILE: &str = "watch.log";

#[derive(Debug, Clone)]
pub(crate) struct WatchLogger {
    path: PathBuf,
}

impl WatchLogger {
    pub(crate) fn new(cache_dir: &Path) -> Self {
        Self {
            path: log_file_path(cache_dir),
        }
    }

    #[cfg(test)]
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn write_line(&self, line: &str) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        self.rotate_if_needed(line.len() as u64 + 1)?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        Ok(())
    }

    fn rotate_if_needed(&self, incoming_bytes: u64) -> Result<()> {
        let current_size = match fs::metadata(&self.path) {
            Ok(metadata) => metadata.len(),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(err.into()),
        };

        if current_size.saturating_add(incoming_bytes) <= WATCH_LOG_MAX_BYTES {
            return Ok(());
        }

        for index in (1..=WATCH_LOG_ROTATIONS).rev() {
            let source = rotated_path(&self.path, index);
            let target = rotated_path(&self.path, index + 1);
            if source.exists() {
                if index == WATCH_LOG_ROTATIONS {
                    fs::remove_file(&source)?;
                } else {
                    fs::rename(source, target)?;
                }
            }
        }

        if self.path.exists() {
            fs::rename(&self.path, rotated_path(&self.path, 1))?;
        }

        Ok(())
    }
}

pub(crate) fn log_file_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join(LOG_DIR).join(WATCH_LOG_FILE)
}

pub(crate) fn read_recent_lines(path: &Path, max_lines: usize) -> Result<String> {
    let mut file = match OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(String::new()),
        Err(err) => return Err(err.into()),
    };
    let len = file.metadata()?.len();
    let read_len = len.min(256 * 1024);
    file.seek(SeekFrom::Start(len.saturating_sub(read_len)))?;
    let mut buffer = String::new();
    file.read_to_string(&mut buffer)?;

    let lines = buffer.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    Ok(lines[start..].join("\n"))
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(WATCH_LOG_FILE);
    path.with_file_name(format!("{name}.{index}"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{log_file_path, read_recent_lines, WatchLogger};

    #[test]
    fn read_recent_lines_returns_tail() {
        let tmp = tempdir().expect("tempdir");
        let path = log_file_path(tmp.path());
        fs::create_dir_all(path.parent().expect("parent")).expect("create log dir");
        fs::write(&path, "one\ntwo\nthree\n").expect("write log");

        let recent = read_recent_lines(&path, 2).expect("read recent");

        assert_eq!(recent, "two\nthree");
    }

    #[test]
    fn logger_writes_log_file() {
        let tmp = tempdir().expect("tempdir");
        let logger = WatchLogger::new(tmp.path());

        logger.write_line("hello").expect("write");

        let content = fs::read_to_string(logger.path()).expect("read");
        assert_eq!(content, "hello\n");
    }
}
