# Security Policy

## Supported versions

Security fixes are made on the latest release and on `master`. Older releases are not patched.

## Reporting a vulnerability

Please report suspected vulnerabilities privately, not in a public issue or pull request.

Use GitHub's private vulnerability reporting: open the repository's **Security** tab and choose **Report a vulnerability**
(<https://github.com/coletebou/ai-cli/security/advisories/new>). Include:

- the `ai --version` output,
- your OS and terminal,
- the steps or input that trigger the problem and what you expect an attacker to gain,
- any suggested fix.

Never include a live API key or token in a report. If a key may have leaked, revoke it with its provider first.

You can expect an acknowledgement within a few days. Fixes are released as soon as they are ready and credited in the
release notes unless you ask otherwise.

## Scope

In scope: how `ai` handles credentials (config files, environment variables, Bedrock bearer tokens, the Anthropic
subscription token file), what it logs or prints (including telemetry exports and `ai doctor` output), how it
expands `@file` references and piped input, and the release and install scripts.

Out of scope: vulnerabilities in the model providers themselves, and issues that need an attacker to already control your
account, shell or config file.
