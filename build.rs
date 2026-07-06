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

    // Rerun if git HEAD changes. HEAD itself only changes on branch switch;
    // commits on the current branch update the ref file it points to, so
    // track that too or the embedded hash goes stale.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD") {
        if let Some(ref_path) = head.strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed=.git/{}", ref_path.trim());
        }
    }
}
