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
- **Interactive chat.** `-i` keeps conversation history so you can ask follow-ups. It supports multi-line input, `@file` references and input history that persists across sessions. Chats can be saved, reloaded and resumed with `-i --continue`. `/clear`, `/history`, `/export`, `/save`, `/load`, `/sessions`, `/help` built in.
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
ai -i --continue                                              # resume the last chat
ai -w "what changed in React 19.1?"                           # web search
ai -r high "prove this theorem"                               # high-reasoning mode
ai --raw "generate a commit message" | pbcopy                 # raw/pipeable
```

### Context expansion

Everything that gets inlined is announced on stderr, e.g. `[context] inlined src/main.rs (12.3 KB)`.

- **Piped stdin.** When stdin is a pipe or a redirected file (`cmd | ai "…"`, `ai "…" < log.txt`), its content is appended to the question as a fenced context block. Anything else (a terminal, `/dev/null`, a socket such as the one Node's `spawn()` hands its children) is never read, so `ai` doesn't sit waiting for input that isn't coming. Stdin over 20 MB is an error; bytes that aren't valid UTF-8 are replaced (with a warning) rather than dropped.
  - Pass `--no-stdin` when a parent process holds stdin open without writing to it (e.g. Python's `Popen(stdin=PIPE)`), since that looks exactly like `cmd | ai`.
  - In shell loops, keep `ai` off the loop's input, or the first call reads the rest of the file as context:
    ```bash
    while read -r q; do ai "$q" </dev/null; done < questions.txt   # or: ai --no-stdin "$q"
    ```
- **`@path` references.** A token starting with `@` at a word boundary (or right after `(`, `[`, `"` or `'`) is replaced with the file's contents inside a fenced code block, using the path's extension as the fence language. The fence grows to outlast any backticks in the file, so a README with its own code blocks stays in one piece.
  - `@~/notes.md` expands the home directory; `@"my notes.md"` quotes a path with spaces. That works as a separate shell word (`ai read @"my notes.md"`) and inside a quoted question (`ai 'read @"my notes.md" please'`). Trailing punctuation stays in the sentence: `(see @notes.md).`
  - A path that doesn't exist passes through as literal text, with a warning if the token looks like a path (contains `/` or `.`). Directories, files over 20 MB, and binary or non-UTF-8 files are errors. UTF-8 with a BOM and UTF-16 with a BOM (what Windows PowerShell 5.1's `>` writes) are decoded.
  - Write `\@notes.md` to send a literal `@notes.md`.
  - `--no-expand` turns expansion off and sends the question exactly as typed. Use it whenever the question includes text you don't control, e.g. `ai --no-expand "Summarize this issue: $TITLE"`. Otherwise an `@/path/to/secret` in that text would be read and sent to the provider.

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

### Cost estimate

When a response reports its token counts, the footer shows an estimated cost, e.g. `cost: $0.0042`, and a multi-provider run ends with a `total: $…` line. The estimate uses built-in list prices (USD per million tokens) for the default Anthropic, OpenAI, Google, and xAI models, matched on the model id regardless of date suffix (`claude-haiku-4-5-20251001`), Bedrock spelling (`us.anthropic.claude-opus-4-7`), or OpenRouter route (`anthropic/claude-opus-4-8`). Models without a known price show no cost.

It's a ballpark figure. It ignores prompt-cache discounts, batch pricing, long-context tiers, and regional-endpoint premiums, and built-in prices can go stale, so check your provider's dashboard for real billing. Add or correct prices in the config:

```toml
[footer]
show_cost = true          # default; `--no-cost` hides it for one run

# USD per million tokens, keyed by model id.
[pricing."anthropic/claude-opus-4-8"]   # this OpenRouter route only
input_per_mtok = 5.0
output_per_mtok = 25.0

[pricing."qwen3:8b"]                    # a local Ollama model: free
input_per_mtok = 0.0
output_per_mtok = 0.0
```

An exact model id takes precedence. After that, a bare id such as `"claude-opus-4-8"` also covers the model's dated, Bedrock, and OpenRouter spellings. The built-in table is the fallback.

### Retry behavior

Transient provider errors (HTTP 408, 425, 429, 500, 502, 503, 504, Anthropic's 529 "overloaded", and connection failures/timeouts) are retried automatically with exponential backoff, for streaming and non-streaming requests and for Bedrock. A streaming request is only retried if it fails before any output has been shown. A `Retry-After` header from the server takes precedence over the default backoff.

```toml
[retry]
max_attempts = 3       # total attempts including the first; 1 disables retry
base_ms = 250          # first backoff delay; doubles each subsequent retry
max_delay_ms = 5000    # ceiling per attempt
```

### Telemetry (optional)

Traces can be exported over OpenTelemetry to a [Langfuse](https://langfuse.com) instance. It is off unless all three keys are set:

| Env var                    | Meaning                                                        |
|----------------------------|----------------------------------------------------------------|
| `LANGFUSE_HOST`            | Base URL, e.g. `https://langfuse.example.com`                  |
| `LANGFUSE_PUBLIC_KEY`      | Project public key                                             |
| `LANGFUSE_SECRET_KEY`      | Project secret key                                             |
| `LANGFUSE_ENABLED`         | `false` (or `0`) turns telemetry off even when keys are set    |
| `LANGFUSE_CAPTURE_CONTENT` | `false` (or `0`) sends metadata only: no prompt or answer text |

By default, traces include the question, message history and answers, and that includes any `@file` contents and piped stdin. Each text attribute is capped at 32 KiB and marked `…[truncated N bytes]`. With `LANGFUSE_CAPTURE_CONTENT=false`, only the model, token counts, timings and errors are sent. Calls to Bedrock-hosted models aren't traced yet.

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
ai -i                               # start empty
ai -i "initial question"            # start with a seed question
cat error.log | ai -i "what broke?" # seed with piped input, then keep chatting
ai -i --continue                    # resume the most recently saved session (short: ai -ic)
ai -ic "one more thing"             # resume it and ask a follow-up straight away
```

With piped stdin, the piped content becomes part of the first question and chat keeps reading your replies from the terminal. Without a terminal (e.g. in CI), `-i` exits with an error; drop `-i` for a one-shot answer.

Slash commands inside chat:

| Command                | Action                                                    |
|------------------------|-----------------------------------------------------------|
| `/exit`, `/q`          | Quit                                                      |
| `/clear`, `/reset`     | Clear conversation history                                |
| `/history`             | Print the full message history                            |
| `/export [path.md]`    | Save conversation as markdown. If you omit the path, it's auto-named `chat-<timestamp>.md` (with `-1`, `-2`… if taken). It never overwrites an existing file. |
| `/export! <path.md>`   | Same, but replace the file if it exists                   |
| `/save [name]`         | Save the chat as a session. Without a name it keeps the session's current name, or gets a timestamp name like `chat-20260930-142501` (UTC). |
| `/save! <name>`        | Same, but replace a different session that has this name  |
| `/load <name>`         | Replace the chat with a saved session                     |
| `/sessions`            | List saved sessions, newest first, with age, turn count, model and first message |
| `/help`                | Show available commands                                   |

A line that starts with a path, such as `/etc/nginx/nginx.conf fails with …`, is sent as a message, not treated as a command. The export path may be quoted (`/export "my chat.md"`) and may start with `~/`.

Sessions:

- **Autosave.** On exit (`/exit` or `Ctrl+D`), a chat with new turns is saved: under its session name if it was saved or loaded before, otherwise as `last`. The next unnamed chat replaces `last`, so use `/save <name>` to keep one. `ai -i --continue` (`-c`, needs `-i`) resumes whichever session was saved most recently.
- **Loading.** `/load` and `--continue` keep the current provider and model. If the session was saved with different ones, chat says so; pick another with `-p`. The session keeps its own system prompt. If the chat you're leaving has unsaved turns, `/load` saves them first (under its name, or a timestamp name) and tells you where.
- **Names** may use letters, digits, `.`, `_` and `-`. `/save <name>` won't replace a different session with the same name unless you use `/save!`. `/clear` starts a new, unnamed chat and leaves the saved session alone.
- **Storage.** Each session is a JSON file in `<data dir>/ai-cli/chats/<name>.json` (same data dir as the input history below). It holds the full message history, including `@file` contents, piped input and web-search results, plus the provider, model and timestamps. Writes are atomic. On Linux and macOS the files are readable only by you (mode 0600).

Input:

- **Multi-line messages.** End a line with a single `\` and press Enter to continue on the next line. The continuation backslashes are removed when the message is sent. Pasting multi-line text keeps it as one message (bracketed paste); press Enter to send it.
- **`@path` references** expand to the file's contents, the same as on the command line.
- **Input history.** Up/Down recall earlier messages, including ones from previous sessions. They're saved in `<data dir>/ai-cli/chat_history.txt`: `~/.local/share` on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows. The file keeps the last 1000 entries. On Linux and macOS it's readable only by you (mode 0600).

`Ctrl+D` exits. At the prompt, `Ctrl+C` clears the current input line. During a reply, `Ctrl+C` stops the streaming response; press it again if the provider doesn't stop, and chat quits (exit code 130, without the autosave). The conversation persists across turns; use `/clear` to reset.

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
| `--no-cost`                 | Hide the estimated cost (overrides config)        |
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
├── pricing.rs    Per-model prices and the footer's cost estimate
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
