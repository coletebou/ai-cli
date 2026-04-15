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

## Before you submit a PR

1. **Build clean:** `cargo build` with no warnings.
2. **Lint:** `cargo clippy -- -D warnings`.
3. **Format:** `cargo fmt`.
4. **Test:** `cargo test`.
5. **Smoke-test the feature you changed:**
   - CLI streaming: `cargo run -- "test question"`
   - TUI: `cargo run -- -t "test question"`
   - Chat: `cargo run -- -i`
   - Search: `cargo run -- -w "what is the latest tokio version?"`

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

- `ai --version` output
- OS / terminal emulator
- Exact command that reproduced the bug
- Expected vs. actual behavior
- Any relevant error output (with API keys scrubbed)

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
