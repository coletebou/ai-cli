# AI-CLI Implementation Plan

Multi-provider AI CLI in Rust — query Grok, ChatGPT, Gemini, Claude simultaneously.

---

## Confirmed Specifications

| Aspect | Decision |
|--------|----------|
| **Binary name** | `ai` |
| **Default behavior** | Query all configured providers (configurable in settings) |
| **Streaming** | Enabled by default |
| **MVP display** | Sequential with clear separators |
| **Conversation mode** | Single question only (no REPL for MVP) |
| **Config location** | `~/.config/ai-cli/config.toml` |
| **Scripting support** | `--raw` flag for pipeable output |
| **Error handling** | Show error inline, continue with other providers |
| **System prompt** | `--system` flag + config file support |
| **Token tracking** | Show tokens used per response |

### Core Providers (MVP)
1. **Claude** (Anthropic)
2. **ChatGPT** (OpenAI)
3. **Grok** (xAI)
4. **Gemini** (Google)

### Additional Providers
5. **Ollama** (local models — zero latency, no API key)
6. **OpenRouter** (unified gateway to 200+ models via OpenAI-compatible API)

---

## Research Summary

### Key Library Choice: `genai` (rust-genai)

**Why this is the winning choice:**
- Already supports ALL target providers: OpenAI, Anthropic (Claude), Gemini, xAI (Grok)
- Also supports: Ollama, Cohere, and OpenRouter (via ServiceTargetResolver for OpenAI-compatible endpoints)
- Normalized API across all providers — one interface for everything
- Native streaming support via `exec_chat_stream()`
- 9,280 downloads/month, actively maintained (v0.5.0 released Jan 2026)
- Handles auth, model mapping, and provider quirks internally

**Alternative considered:** Individual SDKs per provider (`xai-sdk`, `claude-sdk-rs`, etc.) — rejected because `genai` already abstracts this cleanly.

### TUI Library: `ratatui`

- De facto standard for Rust TUIs (successor to tui-rs)
- Constraint-based layouts perfect for side-by-side panels
- Sub-millisecond rendering for smooth streaming display
- Works with crossterm (cross-platform terminal backend)

### Core Dependencies

| Crate | Purpose |
|-------|---------|
| `genai` | Multi-provider AI client |
| `ratatui` | Terminal UI |
| `crossterm` | Terminal backend |
| `clap` | CLI argument parsing |
| `tokio` | Async runtime |
| `futures` | `join_all` for concurrent requests |
| `anyhow` | Error handling |
| `serde` | Config serialization |
| `toml` | Config file format |

---

## Architecture

```
ai-cli/
├── src/
│   ├── main.rs           # Entry point, CLI parsing
│   ├── config.rs         # Config file handling (~/.config/ai-cli/config.toml)
│   ├── providers.rs      # Provider abstraction over genai
│   ├── query.rs          # Query execution (single/multi/streaming)
│   ├── ui/
│   │   ├── mod.rs        # UI module exports
│   │   ├── app.rs        # App state management
│   │   ├── sequential.rs # One-after-another display
│   │   └── sidebyside.rs # Side-by-side panel layout
│   └── display.rs        # Simple non-TUI output (for piping)
├── Cargo.toml
└── config.example.toml
```

### Data Flow

```
User Input (CLI args)
       │
       ▼
┌─────────────────┐
│  Config Loader  │  ← ~/.config/ai-cli/config.toml
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Provider Setup  │  ← genai Client with auth
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Query Executor                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │
│  │Grok │ │GPT  │ │Claude│ │Gemini│  │  ← tokio::join_all (concurrent)
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘   │
│     │       │       │       │       │
│     └───────┴───────┴───────┘       │
│              │                      │
└──────────────┼──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        Display Mode                 │
│  [Sequential] [Side-by-Side] [Raw]  │
└─────────────────────────────────────┘
```

---

## CLI Interface Design

### Basic Usage

```bash
# Query all configured providers (default)
ai "What is the capital of France?"

# Single provider
ai "Quick question" -p claude

# Multiple specific providers
ai "Compare these models" -p grok,claude,gemini

# With system prompt
ai "Explain recursion" --system "You are a patient teacher for beginners"

# Raw output for piping
ai "Give me a bash one-liner to find large files" --raw | bash

# Disable streaming (wait for complete responses)
ai "Long question" --no-stream

# Hide token stats
ai "Simple question" --no-tokens
```

### Example Output (Sequential Mode)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CLAUDE                                              [streaming]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The capital of France is Paris...

                                         tokens: 42 in / 18 out

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CHATGPT                                             [streaming]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Paris is the capital of France...

                                         tokens: 38 in / 22 out

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GROK                                                    [error]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Error: Rate limited. Try again in 30 seconds.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GEMINI                                              [streaming]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
France's capital city is Paris...

                                         tokens: 40 in / 20 out
```

### Provider Aliases

| Short | Full | Model |
|-------|------|-------|
| `grok` | xAI | grok-2 |
| `chatgpt` / `gpt` | OpenAI | gpt-4o |
| `claude` | Anthropic | claude-3-5-sonnet |
| `gemini` | Google | gemini-2.0-flash |

### CLI Structure (clap)

```rust
#[derive(Parser)]
#[command(name = "ai")]
#[command(about = "Query multiple AI providers simultaneously")]
struct Cli {
    /// The question to ask
    question: String,

    /// Providers to query (comma-separated)
    #[arg(short, long, value_delimiter = ',')]
    providers: Option<Vec<Provider>>,

    /// Query all configured providers (default behavior)
    #[arg(short, long)]
    all: bool,

    /// System prompt / persona for all providers
    #[arg(short, long)]
    system: Option<String>,

    /// Raw output (no formatting, pipeable)
    #[arg(long)]
    raw: bool,

    /// Disable streaming (wait for complete responses)
    #[arg(long)]
    no_stream: bool,

    /// Model override for all providers (e.g., gpt-4-turbo)
    #[arg(short, long)]
    model: Option<String>,

    /// Hide token usage statistics
    #[arg(long)]
    no_tokens: bool,
}
```

---

## Config File

Location: `~/.config/ai-cli/config.toml`

```toml
[default]
# Providers to query when none specified (empty = all enabled)
providers = ["claude", "chatgpt", "grok", "gemini"]
stream = true
show_tokens = true

# Optional default system prompt
# system_prompt = "You are a helpful assistant. Be concise."

# Core providers (enabled by default if API key exists)
[providers.anthropic]
enabled = true
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-sonnet-4-20250514"
aliases = ["claude"]
color = "magenta"

[providers.openai]
enabled = true
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o"
aliases = ["chatgpt", "gpt"]
color = "green"

[providers.xai]
enabled = true
api_key_env = "XAI_API_KEY"
default_model = "grok-2"
aliases = ["grok"]
color = "blue"

[providers.google]
enabled = true
api_key_env = "GEMINI_API_KEY"
default_model = "gemini-2.0-flash"
aliases = ["gemini"]
color = "cyan"

# Additional providers (disabled by default)
[providers.openrouter]
enabled = false
api_key_env = "OPENROUTER_API_KEY"
fast_model = "google/gemini-2.5-flash"
smart_model = "anthropic/claude-opus-4-6"
aliases = ["openrouter", "or"]
color = "yellow"
base_url = "https://openrouter.ai/api/v1"

[providers.ollama]
enabled = false
base_url = "http://localhost:11434"
fast_model = "qwen3:8b"
smart_model = "qwen3:14b"
aliases = ["ollama", "local"]
color = "red"
```

---

## Implementation Phases

### Phase 1: MVP (Core Functionality)

**Goal:** Query single/multiple providers, display results sequentially.

1. **Project setup**
   - `cargo init`
   - Add dependencies
   - Basic CLI with clap

2. **Config system**
   - Load from `~/.config/ai-cli/config.toml`
   - Fall back to env vars
   - First-run setup wizard (optional)

3. **Provider abstraction**
   - Wrapper around `genai::Client`
   - Model/alias mapping
   - Error handling per provider

4. **Query execution**
   - Single provider query
   - Multi-provider concurrent query via `join_all`
   - Basic streaming support

5. **Sequential display**
   - Print responses as they complete
   - Color-coded by provider
   - Markdown rendering (optional, via `termimad`)

### Phase 2: TUI Enhancement

**Goal:** Side-by-side display with ratatui.

1. **TUI scaffolding**
   - App state management
   - Event loop (keyboard input)
   - Graceful exit (Ctrl+C, q)

2. **Side-by-side layout**
   - Dynamic panel count based on provider selection
   - Scrollable panels
   - Live streaming updates per panel

3. **Visual polish**
   - Provider logos/colors
   - Loading spinners
   - Response metadata (tokens, latency)

### Phase 3: Future Enhancements (Post-MVP)

- **Tab view** — Switch between providers with keyboard
- **Conversation mode** — Multi-turn chat with context
- **History** — Save/recall previous queries
- **Clipboard** — Copy response to clipboard
- **Export** — Save responses to file (markdown, JSON)
- **Cost tracking** — Estimate API costs
- **Local models** — Ollama integration enabled by default

---

## Key Code Patterns

### Concurrent Provider Queries

```rust
use futures::future::join_all;
use genai::Client;

async fn query_all(
    client: &Client,
    providers: &[ProviderConfig],
    question: &str,
) -> Vec<ProviderResult> {
    let futures = providers.iter().map(|p| {
        let client = client.clone();
        let model = p.model.clone();
        let question = question.to_string();
        async move {
            let request = ChatRequest::new(vec![
                ChatMessage::user(&question),
            ]);
            match client.exec_chat(&model, request, None).await {
                Ok(response) => ProviderResult::Success {
                    provider: p.name.clone(),
                    text: response.first_text().unwrap_or_default(),
                },
                Err(e) => ProviderResult::Error {
                    provider: p.name.clone(),
                    error: e.to_string(),
                },
            }
        }
    });

    join_all(futures).await
}
```

### Streaming with TUI Updates

```rust
use tokio::sync::mpsc;

enum UiMessage {
    Chunk { provider: String, text: String },
    Done { provider: String },
    Error { provider: String, error: String },
}

async fn stream_to_ui(
    client: &Client,
    provider: &ProviderConfig,
    question: &str,
    tx: mpsc::Sender<UiMessage>,
) {
    let request = ChatRequest::new(vec![ChatMessage::user(question)]);

    match client.exec_chat_stream(&provider.model, request, None).await {
        Ok(mut stream) => {
            while let Some(chunk) = stream.next().await {
                if let Ok(event) = chunk {
                    if let Some(text) = event.content_text() {
                        let _ = tx.send(UiMessage::Chunk {
                            provider: provider.name.clone(),
                            text: text.to_string(),
                        }).await;
                    }
                }
            }
            let _ = tx.send(UiMessage::Done {
                provider: provider.name.clone(),
            }).await;
        }
        Err(e) => {
            let _ = tx.send(UiMessage::Error {
                provider: provider.name.clone(),
                error: e.to_string(),
            }).await;
        }
    }
}
```

### Side-by-Side Layout

```rust
use ratatui::{
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Wrap},
    Frame,
};

fn render_side_by_side(frame: &mut Frame, responses: &[ProviderResponse]) {
    let num_panels = responses.len();
    let constraints: Vec<Constraint> = (0..num_panels)
        .map(|_| Constraint::Percentage(100 / num_panels as u16))
        .collect();

    let areas = Layout::horizontal(constraints).split(frame.area());

    for (i, response) in responses.iter().enumerate() {
        let block = Block::bordered()
            .title(response.provider.to_uppercase())
            .border_style(response.color());

        let paragraph = Paragraph::new(response.text.as_str())
            .block(block)
            .wrap(Wrap { trim: true });

        frame.render_widget(paragraph, areas[i]);
    }
}
```

---

## Environment Variables

Required API keys (set in shell or `.env`):

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export XAI_API_KEY="xai-..."
export GEMINI_API_KEY="..."
```

---

## Testing Strategy

1. **Unit tests** — Config parsing, provider mapping
2. **Integration tests** — Mock API responses
3. **Manual testing** — Real API calls (use cheap models)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| API rate limits | Configurable delay between requests |
| Provider downtime | Timeout + graceful error display |
| Streaming quirks | Provider-specific handling in genai |
| Large responses | Scrollable panels, truncation option |

---

## Estimated Cargo.toml

```toml
[package]
name = "ai-cli"
version = "0.1.0"
edition = "2021"

[dependencies]
genai = "0.5"
ratatui = "0.29"
crossterm = "0.28"
clap = { version = "4", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
futures = "0.3"
anyhow = "1"
serde = { version = "1", features = ["derive"] }
toml = "0.8"
dirs = "5"  # For ~/.config path
colored = "2"  # For non-TUI colored output
```

---

## Next Steps

1. Review and approve this plan
2. Initialize Cargo project with dependencies
3. Implement Phase 1 (MVP) in order:
   - CLI parsing
   - Config loading
   - Single provider query
   - Multi-provider concurrent query
   - Sequential display
4. Test with real API keys
5. Iterate on Phase 2 (TUI)

---

## Sources

- [rust-genai](https://github.com/jeremychone/rust-genai) — Multi-provider AI client
- [genai docs](https://docs.rs/genai) — API documentation
- [ratatui](https://ratatui.rs/) — Terminal UI framework
- [clap](https://docs.rs/clap) — CLI argument parser
- [tokio join_all](https://rust-lang.github.io/async-book/06_multiple_futures/02_join.html) — Concurrent futures
- [xai-sdk](https://github.com/0xC0DE666/xai-sdk) — Grok SDK (backup option)
