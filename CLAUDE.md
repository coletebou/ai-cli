# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build              # Development build
cargo build --release    # Optimized release build
cargo check              # Fast compilation check
cargo clippy --all-targets --all-features --locked -- -D warnings   # Lint exactly as CI does
cargo fmt --all -- --check                                          # CI also checks formatting
cargo test               # Run all tests
cargo test test_name     # Run specific test
```

## Running

```bash
cargo run -- "question"                    # Query the configured provider(s)
cargo run -- "question" -p claude,gpt      # Specific providers
cargo run -- "question" -t                 # TUI mode (side-by-side panels)
cargo run -- -i                            # Interactive chat
cargo run -- doctor                        # Config / credential / connectivity diagnostics
cargo run -- --init                        # Generate a commented config file
```

The config lives at `~/.config/ai-cli/config.toml` on Linux and macOS
(`$XDG_CONFIG_HOME` and `$AI_CLI_CONFIG` override it; on macOS an existing
`~/Library/Application Support/ai-cli/config.toml` is still used, but only
when there's no config file at the new location)
and at `%APPDATA%\ai-cli\config.toml` on Windows.

## Architecture

Multi-provider AI CLI that queries multiple LLM providers (Claude, GPT, Grok, Gemini, OpenRouter, Ollama, and Claude/OpenAI models on AWS Bedrock) simultaneously and displays responses via streaming terminal output, an interactive chat, or a side-by-side TUI.

### Module Structure

```
src/
├── main.rs                CLI parsing (clap), mode selection, exit status
├── config.rs              Config load/save (path resolution, partial-section merge, atomic 0600 writes)
├── config_ui.rs           `ai config` TUI editor
├── providers.rs           ProviderManager: genai client, model→provider routing, auth, retries, provider selection
├── query.rs               Query execution, tool-use loop, StreamEvent enum (genai and Bedrock paths)
├── retry.rs               Error classification (genai, Bedrock, reqwest) + backoff / Retry-After
├── bedrock.rs             Direct Bedrock Converse/ConverseStream client (bearer token)
├── bedrock_eventstream.rs AWS event-stream frame decoder (CRC-checked)
├── display.rs             CLI output: progressive markdown streaming, multi-provider blocks, --raw
├── pricing.rs             Per-model price table (+ [pricing] overrides) for the footer cost estimate
├── markdown.rs            pulldown-cmark → ratatui styled lines (TUI)
├── ui.rs                  `ai -t`: interactive side-by-side chat panels
├── term.rs                TerminalGuard: restores raw mode / alt screen on every exit path and panic
├── diag.rs                stderr diagnostics, held back while a full-screen UI owns the terminal
├── chat.rs                `ai -i`: rustyline chat with slash commands, history, @file expansion
├── sessions.rs            Chat session files (/save, /load, /sessions, --continue)
├── sessions_cmd.rs        `ai sessions`: table/JSON session listing for scripts and other tools
├── context.rs             Piped stdin and @file expansion into the prompt
├── search.rs              web_search tool: Gemini grounding / Tavily with fallback
├── models.rs              `ai models`: OpenRouter model listing (cached)
├── doctor.rs              `ai doctor`: config, credential and connectivity diagnostics
├── telemetry.rs           Optional OpenTelemetry → Langfuse export
└── anthropic_oauth.rs     auth_mode = "subscription" token loading (Anthropic only)
```

### Data Flow

1. CLI args parsed → config loaded. Missing sections and fields take their defaults, and a `[providers.<name>]` section for a built-in provider is completed from that provider's built-in values. Providers themselves are the exception: a file with no `[providers.*]` sections gets every built-in, but once it has any, only the providers it lists exist (`Config::fill_builtin_provider_fields`).
2. `ProviderManager` created: genai `Client` with custom auth + service-target resolvers (both driven by `ModelRouter`), plus Bedrock clients for providers whose `base_url` is a Bedrock endpoint
3. `select_providers()` picks providers for the query and reports why any were skipped (unknown alias, disabled, no credentials). If none are left, `run()` fails and the process exits 1 in every mode.
4. Dispatch by mode:
   - **One-shot** (default): `stream_all()` (one spawned `stream_single` task per provider + mpsc channel) or `query_all()` (concurrent `join_all` of `query_single`, used for `--no-stream` and raw multi-provider output). `display` renders progressive single-provider streaming, per-provider blocks as each finishes, or raw pipeable output. The process exits 1 if any provider failed.
   - **Chat** (`-i`, `chat::run_interactive`): the first selected provider only; each turn calls `stream_with_messages()` directly with the conversation so far.
   - **TUI** (`-t`, `ui::run_tui`): one spawned `panel_task` per provider, each calling `stream_with_messages()` with its own history.

   Chat and TUI show provider errors in the conversation or panel and exit 0; they exit 1 only if the session itself fails (e.g. no terminal).

### Key Patterns

- **Provider abstraction**: Aliases ("claude", "gpt") resolve to internal names ("anthropic", "openai"). Routing looks at which `[providers.*]` section lists the model first, then genai's guess; `vendor/model` ids not configured anywhere go to OpenRouter.
- **Streaming via channels**: `StreamEvent` (Start / Chunk / SearchStart / SearchEnd / Done / Error) is sent through mpsc and consumed by the CLI display, chat and TUI. Every provider task ends with exactly one Done or Error.
- **Retries**: transient failures (HTTP 408/425/429/500/502/503/504/529 per `is_retryable_status`, connect/timeout, Bedrock throttling and early stream closes) are retried with backoff as long as nothing has been shown to the user yet. genai sends streaming requests lazily, so `exec_chat_stream` polls the first real stream event before handing the stream over.
- **Cancellation**: chat and TUI turns carry a `CancellationToken`; opening a stream, retry backoff, tool calls and every stream read race against it.
- **No stderr while a TUI is up**: use `crate::diag::warn` instead of `eprintln!` for anything that can fire mid-request.
- **Chat history file**: never call rustyline's `FileHistory::save` (or `append` on a file that doesn't exist yet, which falls back to it): it temporarily changes the process-wide umask, which races with other threads creating files. Go through `HistoryFile` in chat.rs, which creates the file itself (`create_private_file`) and compacts via `write_history_file`.
- **Config + CLI override**: Config file sets defaults, CLI flags override (e.g., `--no-tui` overrides `tui: true` in config)

### Provider Configuration

API keys come from the env var named by each provider's `api_key_env` (built-in providers fall back to ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY, OPENROUTER_API_KEY), or from `api_key` in the config. Empty or whitespace-only values count as missing. Bedrock providers use a bearer token (`AWS_BEARER_TOKEN_BEDROCK`). Providers that are disabled or lack credentials are skipped; an explicit `-p` request prints why.

### Dependencies

- **genai**: Unified multi-provider AI client (handles Claude, GPT, Grok, Gemini, etc.)
- **ratatui/crossterm**: TUI framework and terminal backend
- **tokio/futures**: Async runtime and concurrent execution
- **clap**: CLI argument parsing with derive macros
- **opentelemetry / opentelemetry-otlp**: Optional telemetry export to Langfuse

### Observability

`src/telemetry.rs` initializes OpenTelemetry export to a self-hosted Langfuse
when `AI_CLI_LANGFUSE_HOST`, `AI_CLI_LANGFUSE_PUBLIC_KEY`, and
`AI_CLI_LANGFUSE_SECRET_KEY` are set (no-op otherwise; if they are set but the
exporter fails to build, a one-line `[telemetry] disabled: <err>` goes to
stderr). If any of that trio is set, only the trio is used: a partial trio
leaves telemetry off with a one-line warning naming the missing vars (never
values), and never falls back to the generic keys, which would mix projects.
With none of it set, the generic `LANGFUSE_HOST`/`_PUBLIC_KEY`/`_SECRET_KEY`
still work for backward compatibility, but other apps' Langfuse SDKs read
those too, so prefer the `AI_CLI_` names in docs and examples. The precedence
rules live in `telemetry::resolve`, which takes the env lookup as a closure so
tests never mutate the process environment. Each CLI invocation creates a
root `ai.query` span; each genai-routed provider call becomes a child
`chat <PROVIDER>` GENERATION span carrying `gen_ai.*` semconv attributes plus
`langfuse.observation.input/output`. Bedrock-routed calls
(`query_single_bedrock` / `stream_bedrock`) are not traced yet: they create no
span, so only the root `ai.query` span shows up for them. The trace name is the
first 60 chars of the question. Service is identified as `service.name=ai-cli`
so traces are filterable in shared Langfuse projects. Set
`AI_CLI_LANGFUSE_ENABLED=false` as a kill switch; it overrides the generic
`LANGFUSE_ENABLED` when set, so it also turns ai-cli off when the generic
trio belongs to another app.

Prompt/answer text (`langfuse.trace.input`, `langfuse.observation.input`,
`langfuse.observation.output`) must go through `telemetry::content_attr`, which
caps each value at 32 KiB (`MAX_CONTENT_BYTES`, cut on a char boundary with a
`…[truncated N bytes]` marker) so an inlined `@file` or large piped stdin can't
make the OTLP batch fail. Set `AI_CLI_LANGFUSE_CAPTURE_CONTENT=false` (or `0`;
overrides the generic `LANGFUSE_CAPTURE_CONTENT`) to send
metadata only (model, token counts, timings, errors); the trace name then
becomes a generic `query` so it doesn't leak the prompt either.

The root span is ended explicitly when `run()` returns (`EndSpanOnDrop`), not
when its last `Context` drops: detached tasks (`stream_all`, TUI panel workers)
hold clones that can outlive `telemetry::shutdown()`, and a span ending after
shutdown is discarded.
