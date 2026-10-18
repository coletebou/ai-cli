//! Embeds git metadata for `ai --version` and the stale-binary warning in
//! main.rs:
//!
//! - `AI_CLI_GIT_HASH`: full commit hash, or "unknown"
//! - `AI_CLI_GIT_SHORT`: abbreviated commit hash, or "unknown"
//! - `AI_CLI_REPO_PATH`: the checkout the binary was built from, or ""
//!
//! Metadata is only recorded when this crate is the root of its own git
//! checkout. A crates.io tarball or any other copy without `.git` gets
//! "unknown", and a copy that sits inside some *other* repository (vendored
//! into a monorepo, or a home directory kept under git) must not pick up
//! that repository's HEAD, or the stale-binary check would fire on every
//! commit to it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Declaring any rerun-if-changed path turns off cargo's default of
    // re-running on every file change in the package, so list ourselves.
    println!("cargo:rerun-if-changed=build.rs");

    // OK_BECAUSE: cargo always sets CARGO_MANIFEST_DIR for build scripts.
    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );

    let own_checkout = is_own_checkout(&manifest_dir);
    let head = if own_checkout {
        head_commit(&manifest_dir)
    } else {
        None
    };

    // Watch git state whenever this is our own checkout — including a fresh
    // `git init` with no commit yet, so the first commit re-runs this script
    // instead of leaving "unknown" baked in until build.rs itself changes.
    if own_checkout {
        for path in head_watch_paths(&manifest_dir) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    match head {
        Some((full, short)) => {
            println!("cargo:rustc-env=AI_CLI_GIT_HASH={}", full);
            println!("cargo:rustc-env=AI_CLI_GIT_SHORT={}", short);
            println!(
                "cargo:rustc-env=AI_CLI_REPO_PATH={}",
                manifest_dir.display()
            );
        }
        None => {
            println!("cargo:rustc-env=AI_CLI_GIT_HASH=unknown");
            println!("cargo:rustc-env=AI_CLI_GIT_SHORT=unknown");
            println!("cargo:rustc-env=AI_CLI_REPO_PATH=");
        }
    }
}

/// Run `git -C <dir> <args>` and return its trimmed stdout. `None` if git
/// isn't installed, exits non-zero (e.g. "not a git repository"), or prints
/// nothing.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// True when `dir` is the top level of a git worktree, rather than a plain
/// directory or a subdirectory of an unrelated repository. Both sides are
/// canonicalized so symlinks, `C:/` vs `C:\` separators on Windows, and
/// trailing slashes don't cause a mismatch.
fn is_own_checkout(dir: &Path) -> bool {
    let Some(toplevel) = git(dir, &["rev-parse", "--show-toplevel"]) else {
        return false;
    };
    match (std::fs::canonicalize(&toplevel), std::fs::canonicalize(dir)) {
        (Ok(toplevel), Ok(dir)) => toplevel == dir,
        _ => false,
    }
}

/// Full and abbreviated hash of HEAD. `None` in a repository without
/// commits.
fn head_commit(dir: &Path) -> Option<(String, String)> {
    let full = git(dir, &["rev-parse", "--verify", "HEAD"])?;
    let short = git(dir, &["rev-parse", "--short", "--verify", "HEAD"])?;
    Some((full, short))
}

/// Files whose modification means HEAD may now name a different commit.
///
/// Paths come from `git rev-parse --git-path` rather than being hardcoded
/// under `.git/`: in a worktree or submodule `.git` is a file pointing
/// elsewhere, and after `git gc` branch refs live in `packed-refs` instead of
/// loose files. Only existing paths are returned, because cargo treats a
/// missing rerun-if-changed path as always changed and would re-run this
/// script, and rebuild the crate, on every build.
fn head_watch_paths(dir: &Path) -> Vec<PathBuf> {
    // --git-path may print a path relative to `dir`; join() keeps absolute
    // ones as they are.
    let git_path = |name: &str| git(dir, &["rev-parse", "--git-path", name]).map(|p| dir.join(p));

    // HEAD changes on checkout (and on commit when detached); packed-refs
    // changes when a packed branch moves.
    let mut paths: Vec<PathBuf> = ["HEAD", "packed-refs"]
        .iter()
        .filter_map(|name| git_path(name))
        .collect();

    // On a branch, commits rewrite the branch's loose ref file, not HEAD.
    if let Some(branch_ref) = git(dir, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(loose_ref) = git_path(&branch_ref) {
            if loose_ref.is_file() {
                paths.push(loose_ref);
            } else if let Some(refs_dir) = git_path("refs") {
                // The branch is only in packed-refs right now, and the next
                // commit writes it back out as a loose file. Watch the
                // nearest existing directory it will appear under so that
                // commit is still noticed.
                paths.extend(
                    loose_ref
                        .ancestors()
                        .skip(1)
                        .take_while(|p| p.starts_with(&refs_dir))
                        .find(|p| p.is_dir())
                        .map(Path::to_path_buf),
                );
            }
        }
    }

    paths.retain(|p| p.exists());
    paths
}
