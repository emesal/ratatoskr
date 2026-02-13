//! Version information with embedded git metadata.
//!
//! On main: `0.2.0`
//! On other branches: `0.2.0-dev (abc1234 2026-02-13)`
//! Dirty suffix appended when working tree has uncommitted changes.

/// Package version from Cargo.toml.
pub const PKG_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git branch at build time, or "unknown" if unavailable.
pub const GIT_BRANCH: &str = match option_env!("VERGEN_GIT_BRANCH") {
    Some(branch) => branch,
    None => "unknown",
};

/// Git commit SHA at build time, or "unknown" if unavailable.
pub const GIT_SHA: &str = match option_env!("VERGEN_GIT_SHA") {
    Some(sha) => sha,
    None => "unknown",
};

/// Build timestamp (RFC 3339), or "unknown" if unavailable.
pub const BUILD_TIMESTAMP: &str = match option_env!("VERGEN_BUILD_TIMESTAMP") {
    Some(ts) => ts,
    None => "unknown",
};

/// Whether the working tree was dirty at build time.
pub fn git_dirty() -> bool {
    option_env!("VERGEN_GIT_DIRTY") == Some("true")
}

/// Extract just the date (YYYY-MM-DD) from the build timestamp.
fn build_date() -> &'static str {
    // VERGEN_BUILD_TIMESTAMP is RFC 3339: "2026-02-13T..."
    // Take the first 10 characters for the date portion.
    if BUILD_TIMESTAMP.len() >= 10 {
        // const-safe: slice a fixed prefix from a &'static str
        let bytes = BUILD_TIMESTAMP.as_bytes();
        // Safety: RFC 3339 dates are ASCII, so splitting at byte 10 is valid UTF-8.
        match core::str::from_utf8(bytes.split_at(10).0) {
            Ok(date) => date,
            Err(_) => BUILD_TIMESTAMP,
        }
    } else {
        BUILD_TIMESTAMP
    }
}

/// Short (7-char) commit SHA.
fn short_sha() -> &'static str {
    &GIT_SHA[..7.min(GIT_SHA.len())]
}

/// Human-readable version string.
///
/// On main: `0.2.0`
/// On other branches: `0.2.0-dev (abc1234 2026-02-13)`
/// Dirty working tree appends `.dirty` to the SHA.
pub fn version_string() -> String {
    if GIT_BRANCH == "main" {
        PKG_VERSION.to_string()
    } else {
        let dirty = if git_dirty() { ".dirty" } else { "" };
        format!(
            "{PKG_VERSION}-{GIT_BRANCH} ({}{dirty} {})",
            short_sha(),
            build_date()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_starts_with_pkg_version() {
        let version = version_string();
        assert!(
            version.starts_with(PKG_VERSION),
            "version should start with pkg version, got: {version}"
        );
    }

    #[test]
    fn version_string_branch_logic() {
        let version = version_string();
        if GIT_BRANCH == "main" {
            // On main: bare version, no branch/sha/date
            assert_eq!(version, PKG_VERSION);
        } else {
            // On other branches: contains branch name, sha, and date
            assert!(version.contains(GIT_BRANCH), "should contain branch name");
            assert!(version.contains(short_sha()), "should contain short sha");
            assert!(version.contains(build_date()), "should contain build date");
        }
    }

    #[test]
    fn git_sha_is_populated() {
        assert_ne!(GIT_SHA, "unknown", "git SHA should be populated");
    }

    #[test]
    fn git_branch_is_populated() {
        assert_ne!(GIT_BRANCH, "unknown", "git branch should be populated");
    }

    #[test]
    fn build_date_is_ymd() {
        let date = build_date();
        assert!(date.len() >= 10, "build date should be at least 10 chars");
        assert_eq!(&date[4..5], "-", "should have dash at position 4");
        assert_eq!(&date[7..8], "-", "should have dash at position 7");
    }
}
