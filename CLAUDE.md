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
