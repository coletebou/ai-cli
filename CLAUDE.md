# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
cargo build              # Development build
cargo build --release    # Optimized release build
cargo check              # Fast compilation check
cargo clippy             # Lint and suggestions
cargo test               # Run all tests
cargo test test_name     # Run specific test
```

## Running

```bash
cargo run -- "question"                    # Query all default providers
cargo run -- "question" -p claude,gpt      # Specific providers
cargo run -- "question" -t                 # TUI mode (side-by-side panels)
cargo run -- --init                        # Generate config at ~/.config/ai-cli/config.toml
```

## Architecture

Multi-provider AI CLI that queries multiple LLM providers (Claude, GPT, Grok, Gemini, etc.) simultaneously and displays responses via streaming terminal output or interactive TUI.

### Module Structure

```
src/
├── main.rs       Entry point, CLI parsing (clap), mode selection
├── config.rs     Config loading from ~/.config/ai-cli/config.toml, alias resolution
├── providers.rs  ProviderManager wrapping genai::Client, auth resolution
├── query.rs      Query execution (single/concurrent/streaming), StreamEvent enum
├── display.rs    Sequential output formatting, streaming display handler
└── ui.rs         TUI with ratatui - side-by-side panels, keyboard navigation
```

### Data Flow

1. CLI args parsed → Config loaded (with defaults fallback)
2. ProviderManager created with genai::Client + custom AuthResolver
3. Query executed: `query_all()` (concurrent via `join_all`) or `stream_all()` (spawned tasks + mpsc channel)
4. Display: sequential streaming, TUI panels, or raw pipeable output

### Key Patterns

- **Provider abstraction**: Aliases ("claude", "gpt") resolve to internal names ("anthropic", "openai") which map to genai adapters
- **Streaming via channels**: `StreamEvent` enum (Start/Chunk/Done/Error) sent through mpsc, consumed by display/TUI
- **Config + CLI override**: Config file sets defaults, CLI flags override (e.g., `--no-tui` overrides `tui: true` in config)

### Provider Configuration

API keys resolved via environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY) or config file. Providers without keys are automatically filtered out.

### Dependencies

- **genai**: Unified multi-provider AI client (handles Claude, GPT, Grok, Gemini, etc.)
- **ratatui/crossterm**: TUI framework and terminal backend
- **tokio/futures**: Async runtime and concurrent execution
- **clap**: CLI argument parsing with derive macros
- **opentelemetry / opentelemetry-otlp**: Optional telemetry export to Langfuse

### Observability

`src/telemetry.rs` initializes OpenTelemetry export to a self-hosted Langfuse
when `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` are set
(no-op otherwise; if they are set but the exporter fails to build, a one-line
`[telemetry] disabled: <err>` goes to stderr). Each CLI invocation creates a
root `ai.query` span; each genai-routed provider call becomes a child
`chat <PROVIDER>` GENERATION span carrying `gen_ai.*` semconv attributes plus
`langfuse.observation.input/output`. Bedrock-routed calls
(`query_single_bedrock` / `stream_bedrock`) are not traced yet: they create no
span, so only the root `ai.query` span shows up for them. The trace name is the
first 60 chars of the question. Service is identified as `service.name=ai-cli`
so traces are filterable in shared Langfuse projects. Set
`LANGFUSE_ENABLED=false` as a kill switch.

Prompt/answer text (`langfuse.trace.input`, `langfuse.observation.input`,
`langfuse.observation.output`) must go through `telemetry::content_attr`, which
caps each value at 32 KiB (`MAX_CONTENT_BYTES`, cut on a char boundary with a
`…[truncated N bytes]` marker) so an inlined `@file` or large piped stdin can't
make the OTLP batch fail. Set `LANGFUSE_CAPTURE_CONTENT=false` (or `0`) to send
metadata only (model, token counts, timings, errors); the trace name then
becomes a generic `query` so it doesn't leak the prompt either.

The root span is ended explicitly when `run()` returns (`EndSpanOnDrop`), not
when its last `Context` drops: detached tasks (`stream_all`, TUI panel workers)
hold clones that can outlive `telemetry::shutdown()`, and a span ending after
shutdown is discarded.
