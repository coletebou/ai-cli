# ai-cli

A fast, multi-provider AI CLI written in Rust. Query Claude, GPT, Gemini, Grok, and OpenRouter models from one terminal — side-by-side, streaming, with web search and interactive chat.

```
$ ai "explain rust's borrow checker in one paragraph"
$ ai -t "compare these two implementations" -p claude,gpt
$ ai -i                                     # interactive chat
$ ai -w "latest release of tokio?"          # with web search
```

## Highlights

- **One CLI, many providers.** Unified interface over Anthropic, OpenAI, Google, xAI, and OpenRouter via the [`genai`](https://crates.io/crates/genai) crate.
- **Side-by-side TUI.** `-t` splits the terminal into per-provider panels with proper markdown rendering. Interactive by default: type follow-ups, scroll with the mouse wheel, cancel with Ctrl+C.
- **Streaming by default.** Tokens appear as they arrive; no waiting for the full response.
- **Web search tool use.** `-w` exposes a `web_search` tool to any model — backed by Gemini (Google Search grounding) or Tavily, with automatic fallback.
- **Interactive chat.** `-i` keeps conversation history so you can ask follow-ups. `/clear`, `/history`, `/help` built in.
- **Config-driven.** Edit `~/.config/ai-cli/config.toml` or run `ai config` for a TUI editor.
- **Reasoning control.** `-r low|medium|high|<budget>` for models that support it.
- **Raw mode.** `--raw` for pipeable output (good for scripts).

## Install

Requires Rust 1.75+.

```bash
git clone https://github.com/coletebou/ai-cli
cd ai-cli
cargo install --path .
```

This installs `ai` into `~/.cargo/bin` (make sure it's on your `PATH`).

## API keys

Set the env vars for whichever providers you want to use. Providers without keys are silently skipped.

| Provider    | Env var                                  |
|-------------|------------------------------------------|
| Claude      | `ANTHROPIC_API_KEY`                      |
| ChatGPT     | `OPENAI_API_KEY`                         |
| Grok        | `XAI_API_KEY`                            |
| Gemini      | `GEMINI_API_KEY` *(or `GOOGLE_API_KEY`)* |
| OpenRouter  | `OPENROUTER_API_KEY`                     |
| Tavily (web search) | `TAVILY_API_KEY`                 |

## Quick start

```bash
ai "what is the airspeed velocity of an unladen swallow?"    # default providers
cat file.md | ai "summarize this"                             # pipe stdin as context
ai "explain @src/main.rs briefly"                             # inline a file with @path
ai -p claude,gpt "compare these approaches"                   # specific providers
ai -t "design a rate limiter"                                 # TUI mode
ai -i                                                         # interactive chat
ai -w "what changed in React 19.1?"                           # web search
ai -r high "prove this theorem"                               # high-reasoning mode
ai --raw "generate a commit message" | pbcopy                 # raw/pipeable
```

### Context expansion

- **Piped stdin.** When stdin isn't a TTY, its content is appended to the question as a fenced context block.
- **`@path` references.** Any token starting with `@` at a word boundary is replaced with the file's contents inside a fenced code block. The path's extension is used as the fence language. Non-existent paths pass through as literal text, so typos are visible.

## Configuration

On first run, generate a config file:

```bash
ai --init
```

This writes to `~/.config/ai-cli/config.toml` on Linux and macOS, or `%APPDATA%\ai-cli\config.toml` on Windows. The location can be changed:

- `AI_CLI_CONFIG=/path/to/config.toml` uses that file instead.
- If `XDG_CONFIG_HOME` is set, the file is `$XDG_CONFIG_HOME/ai-cli/config.toml`.

Older versions used `~/Library/Application Support/ai-cli/config.toml` on macOS. A config there is still used, as long as there's no file at the new location. To switch, move it to `~/.config/ai-cli/`.

On Linux and macOS the file is created readable only by you (mode 0600), because it can hold an `api_key`.

Or edit visually:

```bash
ai config
```

The config UI has sections for **CLI mode**, **TUI mode**, **Footer display**, and **Search**. Each mode has its own default provider(s) and model tier.

### Search providers

Set the primary search backend in config. The other is used automatically as fallback if the primary fails.

```toml
[search]
provider = "gemini"      # or "tavily"
max_results = 5
max_iterations = 5

[search.gemini]
model = "gemini-3.1-flash-lite-preview"
api_key_env = "GEMINI_API_KEY"
fallback_api_key_env = "GOOGLE_API_KEY"

[search.tavily]
api_key_env = "TAVILY_API_KEY"
```

Override at the CLI: `ai -w --search-provider tavily "query"`.

### Retry behavior

Transient provider errors (HTTP 408, 425, 429, 500, 502, 503, 504, Anthropic's 529 "overloaded", and connection failures/timeouts) are retried automatically with exponential backoff, for streaming and non-streaming requests and for Bedrock. A streaming request is only retried if it fails before any output has been shown. A `Retry-After` header from the server takes precedence over the default backoff.

```toml
[retry]
max_attempts = 3       # total attempts including the first; 1 disables retry
base_ms = 250          # first backoff delay; doubles each subsequent retry
max_delay_ms = 5000    # ceiling per attempt
```

## Subcommands

```bash
ai config                           # TUI config editor
ai models                           # list available OpenRouter models
ai models --free                    # only free models
ai models --refresh                 # force-refresh model cache
ai models "llama"                   # filter by name
ai completions zsh                  # generate shell completion script
```

## Shell completions

```bash
# zsh
ai completions zsh > ~/.zfunc/_ai       # then: fpath+=(~/.zfunc); autoload -Uz compinit && compinit

# bash
ai completions bash > ~/.local/share/bash-completion/completions/ai

# fish
ai completions fish > ~/.config/fish/completions/ai.fish
```

## Interactive chat

```bash
ai -i                    # start empty
ai -i "initial question" # start with a seed question
```

Slash commands inside chat:

| Command                | Action                                                    |
|------------------------|-----------------------------------------------------------|
| `/exit`, `/q`          | Quit                                                      |
| `/clear`, `/reset`     | Clear conversation history                                |
| `/history`             | Print the full message history                            |
| `/export [path.md]`    | Save conversation as markdown (auto-named if path omitted)|
| `/help`                | Show available commands                                   |

`Ctrl+D` exits. `Ctrl+C` cancels the current input line at the prompt, or aborts an in-flight streaming response mid-turn. History persists across turns; use `/clear` to reset.

Chat is single-provider — if multiple are configured, the first is used and the rest are noted as skipped.

## Interactive TUI

```bash
ai -t                      # open empty, type your first question
ai -t "initial question"   # seed the first turn
```

The TUI is a long-running chat: type a message, hit Enter, read responses in each panel, type a follow-up. Each provider keeps its own history, so you can compare how they handle a conversation over multiple turns.

**Keybinds** — vim-style modes:

| Mode | Key | Action |
|------|-----|--------|
| Normal | `i` | Enter insert mode |
| Normal | `h` / `l` / `←` / `→` / `Tab` | Cycle between panels |
| Normal | `j` / `k` / `↑` / `↓` | Scroll selected panel |
| Normal | `PageUp` / `PageDown` | Page-size scroll |
| Normal | `q` / `Esc` | Quit (Esc cancels first if streaming) |
| Insert | `Enter` | Submit turn to all panels |
| Insert | `Esc` | Back to normal |
| Insert | `Ctrl+C` | Cancel in-flight turn, or clear line |
| Insert | `←` / `→` / `Home` / `End` | Move the cursor (also `Ctrl+A` / `Ctrl+E`) |
| Insert | `Ctrl+W` / `Ctrl+U` / `Ctrl+K` | Delete word back / to start / to end |
| Insert | `PageUp` / `PageDown` | Scroll the selected panel |
| Any | Mouse wheel | Scroll the panel under the cursor |

Multi-line pastes are kept intact: newlines show as `↵` in the input box and are sent as real newlines.

Slash commands in the input box: `/clear` (reset all histories), `/exit` (quit).

Mouse capture is enabled so the wheel scrolls panels natively. To text-select in the terminal (which mouse capture normally blocks), hold `Shift` (iTerm2, Terminal.app, most Linux terminals) or `Option`.

## Flags reference

| Flag                        | Description                                       |
|-----------------------------|---------------------------------------------------|
| `-p, --providers <list>`    | Comma-separated providers (`claude,gpt,grok,gemini`) |
| `-a, --all`                 | Query every enabled provider that has credentials |
| `-s, --system <prompt>`     | System prompt / persona                           |
| `-r, --reasoning <level>`   | `none`, `low`, `medium`, `high`, or numeric budget |
| `-w, --search`              | Enable web search tool                            |
| `--search-provider <name>`  | Override search primary (`gemini`, `tavily`)      |
| `-t, --tui`                 | Side-by-side TUI display                          |
| `--no-tui`                  | Disable TUI (overrides config)                    |
| `-i, --interactive`         | Interactive chat mode                             |
| `--raw`                     | No formatting, pipeable                           |
| `--no-stream`               | Wait for complete responses                       |
| `--no-tokens` / `--tokens`  | Hide / show token usage (overrides config)        |
| `--init`                    | Generate default config                           |

Providers that are disabled (`enabled = false`) or have no API key are skipped. When you name one explicitly with `-p`, `ai` prints a note saying why it was skipped; if nothing is left to query, it lists every reason.

**Exit status:** for a one-shot question, `0` when every queried provider answered and `1` if any provider failed or nothing could be queried, so `ai "..." || fallback` works in scripts. Interactive chat (`-i`) and the TUI show provider errors on screen instead: they exit `0` after the session, and `1` only when nothing could be queried or the session itself fails (for example, no terminal).

## Architecture

```
src/
├── main.rs       CLI parsing, mode selection
├── config.rs     TOML config, alias resolution, search settings
├── config_ui.rs  TUI config editor
├── context.rs    @file expansion and stdin context merging
├── providers.rs  ProviderManager wrapping genai::Client + AuthResolver
├── query.rs      Query execution, tool-use loop, StreamEvent enum
├── search.rs     SearchEngine: Tavily + Gemini grounding, fallback logic
├── markdown.rs   pulldown-cmark → ratatui styled lines
├── display.rs    Sequential + streaming terminal output
├── chat.rs       Interactive chat (rustyline) with slash commands
├── models.rs     OpenRouter model listing
└── ui.rs         Ratatui TUI — interactive side-by-side chat panels
```

**Data flow:** CLI → Config → `ProviderManager` → `query_all`/`stream_all` (concurrent) → `Display` / TUI panels.

**Streaming:** per-provider `tokio::spawn` tasks emit `StreamEvent`s through an `mpsc` channel that the display layer consumes.

**Tool use:** when `--search` is set, the model gets a `web_search` tool. The query loop runs up to `max_iterations` turns, executing tool calls and feeding results back until the model returns a final answer.

## Development

```bash
cargo build                # dev build
cargo build --release      # optimized
cargo check                # fast typecheck
cargo clippy               # lint
cargo test                 # run tests
cargo run -- "question"    # run locally
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
