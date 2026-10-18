#!/bin/sh
# Install `ai` from a GitHub release of https://github.com/coletebou/ai-cli
#
#   curl -fsSL https://raw.githubusercontent.com/coletebou/ai-cli/master/install.sh | sh
#
# Picks the archive for this OS and CPU, checks it against the SHA-256
# published next to it, and copies the `ai` binary into ~/.local/bin. It
# never uses sudo, and the only downloaded file it runs is that binary
# (`ai --version`, to check it works here before replacing anything).
#
# Settings, as environment variables for `sh`
# (e.g. `curl -fsSL .../install.sh | AI_VERSION=0.2.0 sh`):
#
#   AI_VERSION       release to install, e.g. 0.2.0 (default: the latest)
#   AI_INSTALL_DIR   directory to install into (default: ~/.local/bin)
#   AI_RELEASES_URL  where to download from, for mirrors and testing
#                    (default: https://github.com/coletebou/ai-cli/releases)

set -eu

say() { printf 'install.sh: %s\n' "$*" >&2; }
die() {
  say "error: $*"
  exit 1
}

releases_url=${AI_RELEASES_URL:-https://github.com/coletebou/ai-cli/releases}
releases_url=${releases_url%/}

# Only ever talk to the scheme we started with, so an https download can't
# be redirected to plain http.
scheme=${releases_url%%://*}
case $scheme in
  http | https) ;;
  *) die "AI_RELEASES_URL must be an http(s) URL, got: $releases_url" ;;
esac
command -v curl > /dev/null 2>&1 || die "curl is required"
fetch() { curl --proto "=$scheme" --tlsv1.2 -fsSL "$@"; }

# --- which build ------------------------------------------------------------

case $(uname -s) in
  Linux) os=unknown-linux-gnu ;;
  Darwin) os=apple-darwin ;;
  MINGW* | MSYS* | CYGWIN*)
    die "on Windows, download the .zip from $releases_url and put ai.exe on your PATH" ;;
  *) die "no prebuilt binary for $(uname -s); build from source instead (see the README)" ;;
esac
case $(uname -m) in
  x86_64 | amd64) arch=x86_64 ;;
  aarch64 | arm64) arch=aarch64 ;;
  *) die "no prebuilt binary for $(uname -m) CPUs; build from source instead (see the README)" ;;
esac
# A shell running under Rosetta reports x86_64 on Apple Silicon; the native
# build is the better choice there.
if [ "$os" = apple-darwin ] && [ "$arch" = x86_64 ] &&
  [ "$(sysctl -n sysctl.proc_translated 2> /dev/null)" = 1 ]; then
  arch=aarch64
fi
# The Linux builds are linked against glibc and won't start on musl
# (Alpine and friends).
if [ "$os" = unknown-linux-gnu ] && ldd --version 2>&1 | grep -qi musl; then
  die "no prebuilt binary for musl-based Linux; build from source instead (see the README)"
fi
target=$arch-$os

if [ -n "${AI_VERSION:-}" ]; then
  version=${AI_VERSION#v}
else
  # .../releases/latest redirects to .../releases/tag/v<version>.
  latest=$(fetch -o /dev/null -w '%{url_effective}' "$releases_url/latest") ||
    die "could not look up the latest release at $releases_url/latest"
  case $latest in
    */tag/v*) version=${latest##*/tag/v} ;;
    *) die "could not work out the latest version from $latest" ;;
  esac
fi
case $version in
  '' | *[!0-9A-Za-z.+-]*) die "not a version number: $version" ;;
esac

# --- download and verify ----------------------------------------------------

name=ai-$version-$target
archive=$name.tar.gz
url=$releases_url/download/v$version/$archive

tmp=$(mktemp -d) || die "could not create a temporary directory"
trap 'rm -rf "$tmp"' EXIT
trap 'exit 1' HUP INT TERM

say "downloading $url"
fetch -o "$tmp/$archive" "$url" ||
  die "download failed; does release v$version have a $target build?"
fetch -o "$tmp/$archive.sha256" "$url.sha256" ||
  die "could not download the checksum $url.sha256"

expected=$(cut -d ' ' -f 1 "$tmp/$archive.sha256")
if command -v sha256sum > /dev/null 2>&1; then
  actual=$(sha256sum "$tmp/$archive" | cut -d ' ' -f 1)
elif command -v shasum > /dev/null 2>&1; then
  actual=$(shasum -a 256 "$tmp/$archive" | cut -d ' ' -f 1)
else
  die "sha256sum or shasum is needed to verify the download"
fi
if [ -z "$actual" ] || [ "$actual" != "$expected" ]; then
  die "checksum mismatch for $archive: expected '$expected', got '$actual'"
fi
say "checksum OK ($actual)"

# Only the binary is taken out of the archive.
tar -xzf "$tmp/$archive" -C "$tmp" "$name/ai" ||
  die "$archive does not contain $name/ai"

# --- install ----------------------------------------------------------------

dir=${AI_INSTALL_DIR:-$HOME/.local/bin}
mkdir -p "$dir" || die "could not create $dir"
[ -w "$dir" ] || die "$dir is not writable; set AI_INSTALL_DIR to a directory you own"
dir=$(cd "$dir" && pwd)
dest=$dir/ai

# Stage the binary next to the destination and check that it runs here
# before touching anything, so a build that can't start (glibc too old, no
# OpenSSL 3) leaves an existing install alone. Staging in $dir rather than
# $tmp also works where /tmp is mounted noexec. Then rename over the old
# binary: the swap is atomic and works even while the old `ai` is running.
staged=$dest.new
cp "$tmp/$name/ai" "$staged"
chmod 755 "$staged"
if ! installed=$("$staged" --version); then
  rm -f "$staged"
  die "the $target build doesn't run on this system (the Linux builds need glibc 2.34+ and OpenSSL 3); nothing was installed"
fi
mv -f "$staged" "$dest"
say "installed $installed to $dest"

case :$PATH: in
  *:"$dir":*)
    found=$(command -v ai 2> /dev/null || true)
    if [ "$found" != "$dest" ]; then
      say "note: $found comes first on your PATH, so \`ai\` still runs that one"
    fi
    ;;
  *)
    say "note: $dir is not on your PATH; add it to your shell profile:"
    say "  export PATH=\"$dir:\$PATH\""
    ;;
esac
