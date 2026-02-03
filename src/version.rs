//! Version information with embedded git metadata.

/// Package version from Cargo.toml.
pub const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git branch at build time, or "unknown" if unavailable.
pub const GIT_BRANCH: &str = match option_env!("VERGEN_GIT_BRANCH") {
    Some(branch) => branch,
    None => "unknown",
};

/// Git commit SHA (short) at build time, or "unknown" if unavailable.
pub const GIT_SHA: &str = match option_env!("VERGEN_GIT_SHA") {
    Some(sha) => sha,
    None => "unknown",
};

/// Whether the working tree was dirty at build time.
pub fn git_dirty() -> bool {
    option_env!("VERGEN_GIT_DIRTY") == Some("true")
}

/// Full version string: `{version}+{branch}.{sha}` or `{version}+{branch}.{sha}.dirty`.
///
/// Examples:
/// - `0.1.0-dev+main.abc1234`
/// - `0.1.0-dev+feature/foo.abc1234.dirty`
pub fn version_string() -> String {
    let dirty_suffix = if git_dirty() { ".dirty" } else { "" };
    format!(
        "{PKG_VERSION}+{GIT_BRANCH}.{}{dirty_suffix}",
        &GIT_SHA[..7.min(GIT_SHA.len())]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_contains_pkg_version() {
        let version = version_string();
        assert!(
            version.starts_with(PKG_VERSION),
            "version should start with pkg version"
        );
    }

    #[test]
    fn version_string_contains_branch() {
        let version = version_string();
        assert!(
            version.contains(GIT_BRANCH),
            "version should contain branch name"
        );
    }

    #[test]
    fn git_sha_is_populated() {
        // SHA should be populated by vergen, not "unknown"
        assert_ne!(GIT_SHA, "unknown", "git SHA should be populated");
    }

    #[test]
    fn git_branch_is_populated() {
        // Branch should be populated by vergen, not "unknown"
        assert_ne!(GIT_BRANCH, "unknown", "git branch should be populated");
    }
}
