use std::process::Command;

fn main() {
    // Get the git commit hash
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok();

    let commit_hash = output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get the repo root path
    let repo_output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok();

    let repo_path = repo_output
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // Export as compile-time environment variables
    println!("cargo:rustc-env=AI_CLI_GIT_HASH={}", commit_hash);
    println!("cargo:rustc-env=AI_CLI_REPO_PATH={}", repo_path);

    // Rerun if git HEAD changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
}
