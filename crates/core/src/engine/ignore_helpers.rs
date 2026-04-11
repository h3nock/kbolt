use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::{Walk, WalkBuilder};
use kbolt_types::KboltError;

use crate::Result;

pub(super) fn is_hard_ignored_dir_name(name: &std::ffi::OsStr) -> bool {
    const HARD_IGNORED_DIRS: &[&str] = &[
        ".git",
        "node_modules",
        "target",
        "dist",
        "build",
        ".next",
        ".turbo",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".cache",
        "coverage",
    ];

    name.to_str().is_some_and(|value| {
        HARD_IGNORED_DIRS
            .iter()
            .any(|ignored| value.eq_ignore_ascii_case(ignored))
    })
}

pub(super) fn is_hard_ignored_file(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) == Some(".DS_Store") {
        return true;
    }

    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lock"))
}

pub(super) fn load_collection_ignore_matcher(
    config_dir: &Path,
    collection_root: &Path,
    space: &str,
    collection: &str,
) -> Result<Option<Gitignore>> {
    let ignore_file = collection_ignore_file_path(config_dir, space, collection);
    if !ignore_file.is_file() {
        return Ok(None);
    }

    let mut builder = GitignoreBuilder::new(collection_root);
    if let Some(err) = builder.add(&ignore_file) {
        return Err(KboltError::InvalidInput(format!(
            "invalid ignore file '{}': {err}",
            ignore_file.display()
        ))
        .into());
    }

    let matcher = builder.build().map_err(|err| {
        KboltError::InvalidInput(format!(
            "invalid ignore file '{}': {err}",
            ignore_file.display()
        ))
    })?;
    Ok(Some(matcher))
}

pub(super) fn build_collection_walk(collection_root: &Path) -> Walk {
    let mut builder = WalkBuilder::new(collection_root);
    builder
        .standard_filters(false)
        .hidden(false)
        .parents(false)
        .ignore(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .require_git(false)
        .follow_links(false)
        .filter_entry(|entry| {
            let is_dir = entry
                .file_type()
                .is_some_and(|file_type| file_type.is_dir());
            !is_dir || !is_hard_ignored_dir_name(entry.file_name())
        });
    builder.build()
}

pub(super) fn collection_ignore_file_path(
    config_dir: &Path,
    space: &str,
    collection: &str,
) -> std::path::PathBuf {
    config_dir
        .join("ignores")
        .join(space)
        .join(format!("{collection}.ignore"))
}

pub(super) fn validate_ignore_pattern(pattern: &str) -> Result<String> {
    if pattern.trim().is_empty() {
        return Err(KboltError::InvalidInput("ignore pattern cannot be empty".to_string()).into());
    }

    if pattern.contains('\n') || pattern.contains('\r') {
        return Err(
            KboltError::InvalidInput("ignore pattern must be a single line".to_string()).into(),
        );
    }

    Ok(pattern.to_string())
}

pub(super) fn count_ignore_patterns(content: &str) -> usize {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .count()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::{build_collection_walk, is_hard_ignored_dir_name};

    #[test]
    fn hard_ignored_dir_names_cover_common_build_artifacts() {
        for name in [
            ".git",
            "node_modules",
            "target",
            "dist",
            "build",
            ".next",
            ".venv",
            "__pycache__",
            "coverage",
        ] {
            assert!(
                is_hard_ignored_dir_name(std::ffi::OsStr::new(name)),
                "expected {name} to be hard ignored"
            );
        }

        assert!(
            !is_hard_ignored_dir_name(std::ffi::OsStr::new("docs")),
            "docs should not be hard ignored"
        );
    }

    #[test]
    fn collection_walk_applies_gitignore_and_hard_ignored_dirs() {
        let root = tempdir().expect("create temp root");
        let collection_root = root.path().join("collection");
        std::fs::create_dir_all(&collection_root).expect("create collection root");
        std::fs::create_dir_all(collection_root.join("src")).expect("create src");
        std::fs::create_dir_all(collection_root.join("nested")).expect("create nested");

        std::fs::write(
            collection_root.join(".gitignore"),
            "ignored.md\nnested/skip.txt\n",
        )
        .expect("write gitignore");
        std::fs::write(collection_root.join("src/lib.rs"), "fn alpha() {}\n").expect("write src");
        std::fs::write(collection_root.join("ignored.md"), "skip me\n")
            .expect("write ignored file");
        std::fs::write(collection_root.join("nested/keep.md"), "keep me\n")
            .expect("write nested keep");
        std::fs::write(collection_root.join("nested/skip.txt"), "skip me too\n")
            .expect("write nested skip");
        std::fs::create_dir_all(collection_root.join("target/debug")).expect("create target");
        std::fs::write(collection_root.join("target/debug/app"), "artifact\n")
            .expect("write target artifact");
        std::fs::create_dir_all(collection_root.join("node_modules/pkg"))
            .expect("create node_modules");
        std::fs::write(
            collection_root.join("node_modules/pkg/index.js"),
            "module.exports = {};\n",
        )
        .expect("write node module");

        let mut walked = Vec::new();
        for entry in build_collection_walk(&collection_root) {
            let entry = entry.expect("walk entry");
            if !entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file())
            {
                continue;
            }

            let relative = entry
                .path()
                .strip_prefix(&collection_root)
                .expect("strip collection root")
                .to_string_lossy()
                .into_owned();
            walked.push(relative);
        }
        walked.sort();

        assert!(walked.contains(&".gitignore".to_string()));
        assert!(walked.contains(&"src/lib.rs".to_string()));
        assert!(walked.contains(&"nested/keep.md".to_string()));
        assert!(!walked.contains(&"ignored.md".to_string()));
        assert!(!walked.contains(&"nested/skip.txt".to_string()));
        assert!(!walked.iter().any(|path| path.starts_with("target/")));
        assert!(!walked.iter().any(|path| path.starts_with("node_modules/")));
    }

    #[test]
    fn collection_walk_does_not_apply_gitignore_from_parent_of_collection_root() {
        let root = tempdir().expect("create temp root");
        let collection_root = root.path().join("workspace").join("collection");
        std::fs::create_dir_all(&collection_root).expect("create collection root");

        std::fs::write(
            root.path().join("workspace").join(".gitignore"),
            "from-parent.md\n",
        )
        .expect("write parent gitignore");
        std::fs::write(collection_root.join(".gitignore"), "from-root.md\n")
            .expect("write collection gitignore");
        std::fs::write(collection_root.join("from-parent.md"), "keep me\n")
            .expect("write parent-controlled file");
        std::fs::write(collection_root.join("from-root.md"), "skip me\n")
            .expect("write root-controlled file");

        let walked = build_collection_walk(&collection_root)
            .map(|entry| entry.expect("walk entry"))
            .filter(|entry| {
                entry
                    .file_type()
                    .is_some_and(|file_type| file_type.is_file())
            })
            .map(|entry| {
                entry
                    .path()
                    .strip_prefix(&collection_root)
                    .expect("strip collection root")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();

        assert!(walked.contains(&"from-parent.md".to_string()));
        assert!(!walked.contains(&"from-root.md".to_string()));
    }
}
