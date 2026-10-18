# Contributing

Thanks for your interest in `ai-cli`. This is a small personal project, but contributions are welcome.

## Development setup

```bash
git clone https://github.com/coletebou/ai-cli
cd ai-cli
cargo build
cargo run -- "hello"
```

At least one provider API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `XAI_API_KEY`, or `OPENROUTER_API_KEY`) must be set for end-to-end testing.

If you install the binary from a checkout (`cargo install --path .`), it warns on startup when the checkout's HEAD has moved past the commit it was built from. Set `AI_CLI_NO_STALE_CHECK=1` to silence that.

## Before you submit a PR

CI runs clippy and the tests on Linux, macOS **and Windows**, so run the same commands locally:

1. **Build clean:** `cargo build` with no warnings.
2. **Lint:** `cargo clippy --all-targets --all-features -- -D warnings`.
3. **Format:** `cargo fmt --all` (CI runs `cargo fmt --all -- --check`).
4. **Test:** `cargo test`.
5. **Smoke-test the feature you changed:**
   - CLI streaming: `cargo run -- "test question"`
   - TUI: `cargo run -- -t "test question"`
   - Chat: `cargo run -- -i`
   - Search: `cargo run -- -w "what is the latest tokio version?"`

### Platforms, MSRV and the lockfile

- **Windows is a CI target.** Paths (`@C:\path` references, the `%APPDATA%` config dir), Ctrl+C handling and rustyline's console backend all behave differently there. Use `std::path`/`dirs` rather than hand-built `/` paths, and if a test genuinely can't run on one platform, gate it with `#[cfg(...)]` and a comment saying why.
- **Minimum supported Rust version** is the `rust-version` in `Cargo.toml` (currently 1.88, set by dependencies). The `msrv` CI job runs `cargo check --all-targets --locked` on exactly that toolchain. To check locally: `rustup toolchain install 1.88 && cargo +1.88 check --all-targets --locked`. If a dependency bump or a newer std API needs a newer Rust, raise `rust-version` in the same PR and say why.
- **CI builds with `--locked`.** If you change dependencies, commit the updated `Cargo.lock`.

## Project layout

See the [Architecture](README.md#architecture) section of the README for an overview of each module.

Key invariants:

- **All providers go through `ProviderManager`.** Don't call `genai::Client` directly from feature code.
- **Streaming uses `StreamEvent`.** New stream-producing code should emit events into the `mpsc::Sender<StreamEvent>` rather than printing directly, so TUI and CLI renderers stay interchangeable.
- **Config has a TUI editor.** When you add a setting to `config.rs`, add a corresponding field to `config_ui.rs` so users can edit it visually.
- **Search backends are pluggable.** New search providers implement the same `SearchProvider` enum pattern in `search.rs` — add a variant, a client struct with `from_env` + `search`, and a branch in `main.rs`'s `build` closure.

## Commit style

- One logical change per commit.
- Conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`) are appreciated but not strictly required.
- Explain the **why** in the commit body if it's not obvious from the diff.

## Reporting bugs

Open a GitHub issue with:

- `ai --version` output (it includes the git commit the binary was built from)
- OS / terminal emulator
- Exact command that reproduced the bug
- Expected vs. actual behavior
- Any relevant error output (with API keys scrubbed)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
