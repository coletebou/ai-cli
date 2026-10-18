# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2021 command-line application. `src/main.rs` owns CLI parsing and
mode selection. Provider configuration and authentication live in
`src/config.rs` and `src/providers.rs`; query orchestration and streaming use
`src/query.rs`. Terminal output, interactive chat, and the TUI are implemented
in `src/display.rs`, `src/chat.rs`, and `src/ui.rs`. Bedrock protocol support is
isolated in `src/bedrock.rs` and `src/bedrock_eventstream.rs`; ChatGPT OAuth and
Bedrock Mantle Responses use `src/openai_responses.rs`. Keep
configuration UI changes synchronized with `src/config_ui.rs`.

Unit tests are colocated in each module under `mod tests`. Binary test data
lives in `tests/fixtures/`. CI configuration is in `.github/workflows/ci.yml`,
and planned work is tracked under `backlog/tasks/`.

## Build, Test, and Development Commands

- `cargo check`: quickly type-check the project.
- `cargo build`: create a development binary.
- `cargo run -- "question"`: query configured default providers.
- `cargo run -- "question" -p claude,gpt`: select providers explicitly.
- `cargo run -- -t "question"`: exercise the interactive TUI.
- `cargo test --all`: run the full test suite.
- `cargo clippy --all-targets --all-features -- -D warnings`: run CI-equivalent
  linting.
- `cargo fmt --all -- --check`: verify formatting without changing files.

Run `cargo run -- --init` to generate the user configuration file.

## Coding Style & Naming Conventions

Use standard `rustfmt` output and four-space indentation. Follow Rust naming:
`snake_case` for functions and modules, `CamelCase` for types and traits, and
`SCREAMING_SNAKE_CASE` for constants. Keep provider access behind
`ProviderManager`; feature code should not call `genai::Client` directly.
Streaming producers should emit `StreamEvent` values instead of printing so
CLI and TUI renderers remain interchangeable.

## Testing Guidelines

Add focused `#[test]` functions beside the code they cover, with descriptive
`snake_case` names. Include regression tests for parsing, provider routing,
stream framing, and UI state transitions. Use fixtures only when realistic
wire data is necessary. Before submitting, run formatting, Clippy, all tests,
and smoke-test the affected CLI or TUI path.

## Commit & Pull Request Guidelines

Prefer one logical change per commit. Recent history follows Conventional
Commits, commonly `feat(scope): ...`, `fix(scope): ...`, and `style: ...`.
Explain non-obvious motivation in the commit body. Pull requests should
describe behavior changes, list validation commands, link relevant issues, and
include terminal output or screenshots when TUI rendering changes.

## Security & Configuration

Never commit API keys or populated configuration files. Use environment
variables such as `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GEMINI_API_KEY`,
or the per-user config under `~/.config/ai-cli/`. Scrub credentials from logs,
fixtures, bug reports, and telemetry examples.
