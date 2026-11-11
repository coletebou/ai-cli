# ai-cli: guide for coding agents

A Rust CLI that queries multiple LLM providers (Claude, GPT, Grok, Gemini, OpenRouter, Ollama, and
Claude/OpenAI on AWS Bedrock) at once, via streaming terminal output, an interactive chat, or a
side-by-side TUI. Humans start at [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md);
this file is what an agent needs on top of them. `CLAUDE.md` only imports this file — edit this one.

## Read before changing behaviour

- [README.md](README.md) "Architecture", "Retry behavior" and "Telemetry (optional)": the module
  map, data flow, and the exact retry/telemetry env-var contract. This file doesn't repeat them.
- [CONTRIBUTING.md](CONTRIBUTING.md) "Before you submit a PR" and "Key invariants": the pre-PR
  checklist and the four invariants every feature PR is expected to hold.
- `PLAN.md` is the original design doc — historical context for early decisions, not current status.

## Hard rules

- **All providers go through `ProviderManager`.** Never call `genai::Client` directly from feature
  code.
- **Streaming uses `StreamEvent`.** New stream-producing code emits events into the
  `mpsc::Sender<StreamEvent>`, never prints directly — CLI, chat and TUI renderers all consume the
  same channel.
- **A new `config.rs` field needs a `config_ui.rs` counterpart** so it's editable from `ai config`,
  not just the TOML file.
- **A new search backend follows `search.rs`'s `SearchProvider` pattern**: a variant, a client
  struct with `from_env` + `search`, a branch in `main.rs`'s `build` closure.
- **Never print or log a secret value.** API keys and the Langfuse trio are named (not valued) in
  warnings (`telemetry::resolve`); prompt/answer content sent to Langfuse goes through
  `telemetry::content_attr`'s 32 KiB cap. Follow the same pattern for any new secret-bearing config.
- **CI builds with `--locked`.** Commit the updated `Cargo.lock` with any dependency change.
- **Windows is a CI target.** Use `std::path`/`dirs`, not hand-built `/` paths; gate a
  platform-specific test with `#[cfg(...)]` and a comment saying why.
- Format with `cargo fmt --all` — CI checks it (`cargo fmt --all -- --check`); this project, unlike
  Cole's JS/TS repos, does use a formatter.
- Never skip, weaken or delete a test to get green; change one only when the intended behavior
  changed.

## Where things are, and what a change can break

| Module(s) | Its changes can break |
|---|---|
| `config.rs`, `config_ui.rs` | settings loading/merging/writing and the `ai config` editor together |
| `providers.rs`, `bedrock.rs`, `bedrock_eventstream.rs`, `retry.rs` | model routing, auth and retry for *every* provider and mode at once |
| `bedrock_catalog.rs` | the TUI model picker's list and `ai models --bedrock` |
| `query.rs` | the tool-use loop and `StreamEvent` production — streaming, non-streaming and the search tool |
| `display.rs`, `markdown.rs`, `ui.rs`, `ui_picker.rs`, `term.rs`, `diag.rs` | CLI rendering, TUI panels and model picker, or terminal-state restoration on exit/panic |
| `chat.rs`, `sessions.rs` | interactive chat (`-i`) and `/save`, `/load`, `/sessions`, `--continue` |
| `sessions_cmd.rs` | `ai sessions`: table/JSON session listing for scripts and other tools |
| `context.rs` | `@file` and piped-stdin expansion into the prompt |
| `search.rs` | the `web_search` tool (Tavily / Gemini grounding, fallback) |
| `pricing.rs` | the footer's cost estimate |
| `models.rs` | `ai models` (OpenRouter listing) |
| `doctor.rs` | `ai doctor` diagnostics |
| `telemetry.rs` | optional Langfuse export, including whether prompt content leaks into traces |
| `anthropic_oauth.rs` | `auth_mode = "subscription"` token loading (Anthropic only) |

## Checks: run what your diff can break, and let CI run the rest

```sh
cargo check                                                          # fast typecheck
cargo clippy --all-targets --all-features --locked -- -D warnings    # lint exactly as CI does
cargo fmt --all -- --check                                           # CI's formatting check
cargo test                                                            # full suite
cargo test test_name                                                 # one test
```

- **The full gate is `.github/workflows/ci.yml`**, on every push to `master` and every pull
  request: `fmt`, `clippy-test` (Linux + macOS matrix), `msrv` (reads `rust-version` from
  `Cargo.toml` and runs `cargo check --all-targets --locked` on exactly that toolchain), and
  `clippy-test-windows` (skipped until a personal Windows runner exists and repo variable
  `PERSONAL_WINDOWS_RUNNER` is `true`). There's no `workflow_dispatch` on `ci.yml` — trigger it by
  pushing a branch or opening a PR, then `gh run watch <id> --exit-status`.
- MSRV locally: `rustup toolchain install 1.88 && cargo +1.88 check --all-targets --locked`. Raise
  `rust-version` in the same PR as any dependency/std bump that needs it, and say why.
- `.github/workflows/release.yml` is separate — it builds and publishes binaries on a `v*` tag push,
  or does a dry run (build + package + `install.sh` smoke test, nothing published) on
  `workflow_dispatch` or a PR touching `release.yml`/`install.sh`. Don't push a `v*` tag unless
  asked; a real release needs the Windows leg too (gated the same way as CI's).

## Running it

```sh
cargo run -- "question"              # query the configured provider(s)
cargo run -- "question" -p claude,gpt   # specific providers
cargo run -- "question" -t            # TUI mode (side-by-side panels)
cargo run -- -i                       # interactive chat
cargo run -- doctor                   # config / credential / connectivity diagnostics
cargo run -- --init                   # generate a commented config file
```

Config lives at `~/.config/ai-cli/config.toml` (`$XDG_CONFIG_HOME`/`$AI_CLI_CONFIG` override it) on
Linux and macOS, `%APPDATA%\ai-cli\config.toml` on Windows. On macOS, an existing
`~/Library/Application Support/ai-cli/config.toml` is still read, but **only** when there's no file
at the new `$XDG_CONFIG_HOME` path — a "wrong config" report can come from either location.

## Known traps

- **Bedrock routing is by provider, not model.** `ProviderManager::bedrock_for` sends every model of a
  Bedrock-backed provider (`Provider::name` is its config key) to Converse, so a model picked at
  runtime works without being listed in the config. Don't key new Bedrock dispatch on the model name.
- **No `eprintln!` while a full-screen UI is up.** Use `crate::diag::warn` for anything that can
  fire mid-request; it's held back while the TUI owns the terminal and flushed after.
- **Never call rustyline's `FileHistory::save`** (or `append` on a file that doesn't exist yet,
  which falls back to `save`) — it temporarily changes the process-wide umask, racing other
  threads' file creation. Go through `HistoryFile` in `chat.rs`, which uses `create_private_file`
  and `write_history_file`.
- `genai` sends streaming requests lazily: `exec_chat_stream` polls the first real stream event
  before handing the stream to its caller — a retry-logic change that skips this poll can start
  retrying a request that already produced output.
- The root telemetry span ends explicitly on `run()`'s return (`EndSpanOnDrop`), not when its last
  `Context` drops: detached tasks (`stream_all`, TUI panel workers) can hold a clone that outlives
  `telemetry::shutdown()`, and a span ending after shutdown is silently discarded.
- `cargo install --path .` warns on startup if the checkout's HEAD has moved past the commit the
  binary was built from (`AI_CLI_NO_STALE_CHECK=1` silences it) — don't reflexively dismiss that
  warning; it can be masking a real "forgot to rebuild" bug.
