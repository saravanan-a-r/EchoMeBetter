"""
Single source of truth for reserved token names (DESIGN.md 5.2, 5.3).

Nothing else in this codebase — or in pretraining / SFT / DPO / inference —
defines these names. They are resolved to integer IDs only after training,
via `sp.piece_to_id(name)`, and frozen into token_map.json. No component may
hardcode a token ID.

pad/eos/unk are NOT listed here: they are assigned via SentencePieceTrainer's
pad_id/eos_id/unk_id arguments, not via user_defined_symbols. bos_id=-1 (no
BOS token) per DESIGN.md 6.

Counts (architecture.md 6.2):
    extra_id sentinels    256
    UL2 mode tokens         3
    style tokens           12
    structural markers      2
    spare reserved         36
    cli reserved            64
    cli vocabulary        896
    ----------------------------
    user_defined_symbols 1269
    (+ pad/eos/unk = 1272 named specials total; with the 256 byte_fallback
    pieces added automatically, 1528 of 33,728 slots are reserved (4.53%),
    leaving 32,200 learned pieces -- unchanged, since vocab_size grew by
    exactly the same 896 the reserved block grew by)
"""

from __future__ import annotations

# Span-corruption sentinels (UL2/T5 style). Pretraining Stage 1 requires
# these; there is no way to add them later without an embedding-table
# resize and a full retrain.
#
# 256, not T5's 100 (architecture.md 6.3). Each masked span consumes one
# uniquely-numbered sentinel, so the requirement is:
#
#     max_encoder_length x corruption_rate / mean_span_length
#     = 2048 x 0.15 / 3 = ~103 for R-denoising at our 2048 context
#
# T5's 100 was sized for a 512-token context; here it would bind at exactly
# 2,000 input tokens -- just below the encoder limit -- and overflow. 256
# gives 2.5x headroom, also covers a 4096 context if that is ever revisited,
# and costs 156 embedding rows (~0.02% of the model).
EXTRA_ID_TOKENS: list[str] = [f"<extra_id_{i}>" for i in range(256)]

# UL2 mixture-of-denoisers mode tokens (architecture.md 7.1).
UL2_MODE_TOKENS: list[str] = ["[R]", "[X]", "[S]"]

# All 12 rephrase styles (DESIGN.md 5.4) -- Phase 1 (trained now) and
# Phase 2 (reserved, data not trained yet). Reserve all 12 now: the cost of
# an unused embedding row is trivial; the cost of adding one later is a
# full retrain.
_PHASE_1_STYLES = ["grammar", "concise", "elaborate", "professional", "friendly"]
_PHASE_2_STYLES = ["assertive", "empathetic", "fluent", "formal", "paraphrase", "polite", "simplify"]
STYLE_TOKENS: list[str] = [f"<style:{name}>" for name in _PHASE_1_STYLES + _PHASE_2_STYLES]

# Structural markers delimiting the span the model must rewrite.
STRUCTURAL_TOKENS: list[str] = ["<text_to_rewrite>", "</text_to_rewrite>"]

# Spare capacity for unforeseen needs (DESIGN.md 5.2). Deliberately NOT
# spent on the sentinel expansion above -- sentinels were provisioned
# properly instead, so this stays genuine emergency capacity.
RESERVED_TOKENS: list[str] = [f"<reserved_{i}>" for i in range(36)]

# Capacity set aside for a second downstream task planned after the rewrite
# model: a CLI helper that turns a plain-English sysadmin question into the
# corresponding shell command. Named distinctly from RESERVED_TOKENS above
# (generic emergency spare) so it is unambiguous, in the vocabulary itself,
# which slots are earmarked for what -- a future Stage 2 fine-tune can start
# assigning these specific IDs a meaning (e.g. a `<task:cli>` mode marker,
# argument-boundary tokens) without colliding with the generic spares or
# guessing which reserved block was meant for which purpose.
#
# Sized at 64 rather than reusing the 36 generic spares: the CLI task's own
# needs (a mode token, plus room to structure command/flag/path syntax) are
# not yet designed in detail, and the cost of reserving too much now is
# trivial (one embedding row each) against the cost of running out later,
# which is a full retrain -- the same "reserve now, use later" reasoning
# RESERVED_TOKENS above and EXTRA_ID_TOKENS's 256 (vs T5's 100) already use.
CLI_TOKENS: list[str] = [f"<cli_reserved_{i}>" for i in range(64)]


# ---------------------------------------------------------------------------
# CLI vocabulary -- 896 pieces of literal command-line text.
#
# Every block above reserves a *name* whose meaning is assigned later. This
# block is the opposite: each entry IS the text it matches, and exists so that
# text stops fragmenting. Measured against the frozen 32,832-piece tokenizer,
# on the five `cli_*` corpus sources: flags cost 231,453 tokens more than they
# need to, paths 125,498, and not one of the 1,500 distinct flags observed was
# a single token. `--verbose` was two pieces, `apt-get` three, `/dev/null`
# four. Each entry here makes its text exactly one.
#
# Stored with SentencePiece's word-boundary marker, U+2581, prepended. A piece
# must carry the boundary in order to replace it; without the marker the
# boundary is emitted as its own token and cancels the saving. Measured on a
# tokenizer built with this project's own trainer settings:
# "ls -la /dev/null" is 10 tokens unaided, 6 with unmarked pieces, 4 with
# these. The marker is not whitespace, so Gate 3 atomicity and the
# no-whitespace-in-a-name rule both still hold.
#
# What is deliberately NOT here: bare command names that are also English
# words or English word-prefixes. Verified empirically -- a piece "cat" splits
# "category" into "cat"+"egory" and "concatenate" into "con"+"cat"+"enate",
# and marking it does not help, because a marked piece still matches the start
# of a marked word: " category" becomes marked-"cat" + "egory" just the same.
# Such a piece would trade the rephrase task's prose fertility (1.31-1.39
# tokens/word, inside its target band) for a CLI saving that does not exist:
# `cat`, `head`, `tail`, `top`, `less`, `tar` and `date` were each measured as
# *already* one token. 229 candidates were rejected on this rule; they are
# named in CLI_VOCABULARY.md so the omission is a recorded decision rather
# than an oversight.
#
# The rule the admitted entries satisfy -- and `tests/unit/test_specials.py`
# enforces -- is that each one either contains a character no English word
# does ('-', '/', '.', '_', a digit), or is a capitalised filename such as
# "Dockerfile", or is a lowercase command name vetted individually against the
# prose corpus.
#
# Sized at 896 so the vocabulary totals 33,728: the reserved block grows by
# exactly what vocab_size grows by, holding learned pieces at 32,200 as every
# block above does, and 33,728 stays divisible by 64 as 32,832 was.

# U+2581, SentencePiece's word-boundary marker. Spelled as an escape rather
# than pasted so it cannot be mistaken for an underscore in review, and kept
# literal rather than imported so this module stays dependency-free.
_SPIECE_UNDERLINE = "\u2581"

# Commands, 219 of them: Ubuntu/Debian and macOS, every one measured as
# more than one token today. Hyphenated and suffixed forms dominate because
# those are what fragment -- `apt-get` was 3 pieces, `ssh-keygen` 4.

_CLI_COMMAND_STRINGS: tuple[str, ...] = (
    "add-apt-repository", "adduser", "afplay", "anacron", "ansible",
    "apt-cache", "apt-get", "apt-key", "apt-mark", "atq", "atrm", "automake",
    "badblocks", "base64", "blkid", "blockdev", "buildah", "bunzip2", "byobu",
    "bzip2", "caffeinate", "cfdisk", "chattr", "chfn", "chgrp", "chpasswd",
    "chsh", "cksum", "codesign", "conda", "containerd", "crictl", "cscope",
    "csplit", "csrutil", "ctags", "deluser", "deno", "depmod", "dirname",
    "diskutil", "dmesg", "dmidecode", "do-release-upgrade", "docker-compose",
    "dpkg-deb", "dpkg-query", "dpkg-reconfigure", "dscl", "dseditgroup",
    "dtrace", "dtruss", "e2fsck", "fgrep", "firewalld", "flatpak", "fs_usage",
    "fuser", "getent", "getfacl", "getopts", "gparted", "gpg2", "groupadd",
    "groupdel", "groupmod", "gunzip", "hdiutil", "hdparm", "hexdump",
    "hostnamectl", "htop", "hwclock", "initctl", "insmod",
    "install_name_tool", "ioreg", "iostat", "ip6tables", "iwconfig", "iwlist",
    "javac", "kextload", "kextstat", "kextunload", "killall", "lastlog",
    "ldconfig", "libtool", "lldb", "localectl", "loginctl", "lpadmin",
    "lpstat", "lsattr", "lsblk", "lscpu", "lshw", "lsmod", "lspci", "lsusb",
    "ltrace", "mawk", "md5sum", "mdfind", "mdimport", "mdls", "mdutil",
    "minikube", "mktemp", "mpstat", "mvn", "ncal", "networksetup", "newgrp",
    "nmap", "nmcli", "nohup", "nproc", "npx", "nsenter", "nvim", "nvram",
    "objdump", "osascript", "otool", "pbcopy", "pbpaste", "pgrep", "pidof",
    "ping6", "pip3", "pkg-config", "pkgutil", "pkill", "plutil", "pmset",
    "pnpm", "podman", "printenv", "pvcreate", "pyenv", "python3", "qlmanage",
    "rbenv", "readelf", "readlink", "realpath", "renice", "resize2fs",
    "rgrep", "rmdir", "rmmod", "rustc", "rustup", "screencapture", "scutil",
    "setfacl", "setopt", "setterm", "sfdisk", "sha1sum", "sha256sum",
    "sha512sum", "shasum", "shopt", "shuf", "skopeo", "smartctl",
    "softwareupdate", "spctl", "ssh-add", "ssh-agent", "ssh-copy-id",
    "ssh-keygen", "stty", "subl", "sw_vers", "swapoff", "swapon",
    "system_profiler", "systemd-analyze", "systemd-run", "textutil",
    "timedatectl", "tldr", "tmutil", "tracepath", "tune2fs", "ubuntu-drivers",
    "ulimit", "unalias", "uname", "unattended-upgrade", "uncompress",
    "unexpand", "unxz", "unzstd", "update-alternatives", "update-grub",
    "update-initramfs", "updatedb", "useradd", "userdel", "usermod",
    "valgrind", "vgcreate", "virtualenv", "visudo", "vmstat", "whereis",
    "xattr", "xcode-select", "xcodebuild", "xcrun", "xxd", "zcat", "zless",
    "zstd",
)

# Paths, config files and dotfiles, 293 of them. `/dev/null` was 4 tokens,
# `/etc/apt/sources.list` 9, `/usr/lib/x86_64-linux-gnu` 17. Includes the
# `~/`-rooted forms and bare config filenames, which appear constantly in
# shell transcripts and man pages.

_CLI_PATH_STRINGS: tuple[str, ...] = (
    ".DS_Store", ".bash_history", ".bash_logout", ".bash_profile", ".bashrc",
    ".cache", ".condarc", ".config/nvim", ".curlrc", ".dockerignore",
    ".editorconfig", ".env", ".gitattributes", ".gitconfig", ".gitignore",
    ".gitmodules", ".htaccess", ".htpasswd", ".inputrc", ".local/bin",
    ".netrc", ".npmrc", ".nvmrc", ".profile", ".python-version",
    ".ruby-version", ".screenrc", ".ssh/authorized_keys", ".ssh/config",
    ".ssh/id_ed25519", ".ssh/id_rsa", ".ssh/known_hosts", ".tmux.conf",
    ".viminfo", ".vimrc", ".wgetrc", ".yarnrc", ".zprofile", ".zshenv",
    ".zshrc", "/Applications", "/Library", "/Library/Extensions",
    "/Library/Frameworks", "/Library/LaunchAgents", "/Library/LaunchDaemons",
    "/Library/Logs", "/Library/Preferences", "/Network", "/System",
    "/System/Applications", "/System/Library",
    "/System/Library/LaunchDaemons", "/Users", "/Users/Guest",
    "/Users/Shared", "/Volumes", "/Volumes/Macintosh", "/bin/bash",
    "/bin/cat", "/bin/csh", "/bin/dash", "/bin/echo", "/bin/ls", "/bin/rm",
    "/bin/sh", "/bin/zsh", "/boot", "/boot/grub", "/dev/console",
    "/dev/disk0", "/dev/disk1", "/dev/fd", "/dev/full", "/dev/loop0",
    "/dev/mapper", "/dev/md0", "/dev/mmcblk0", "/dev/null", "/dev/nvme0n1",
    "/dev/nvme0n1p1", "/dev/ptmx", "/dev/random", "/dev/sda", "/dev/sda1",
    "/dev/sda2", "/dev/sdb", "/dev/sdb1", "/dev/sdc", "/dev/shm",
    "/dev/stderr", "/dev/stdin", "/dev/stdout", "/dev/tty", "/dev/urandom",
    "/dev/vda", "/dev/xvda", "/dev/zero", "/etc/apache2",
    "/etc/apt/preferences", "/etc/apt/sources.list",
    "/etc/apt/sources.list.d", "/etc/bash.bashrc", "/etc/cron.d",
    "/etc/cron.daily", "/etc/crontab", "/etc/default/grub",
    "/etc/environment", "/etc/fstab", "/etc/group", "/etc/hostname",
    "/etc/hosts", "/etc/hosts.allow", "/etc/hosts.deny", "/etc/init.d",
    "/etc/inputrc", "/etc/issue", "/etc/localtime", "/etc/logrotate.conf",
    "/etc/lsb-release", "/etc/machine-id", "/etc/modprobe.d", "/etc/modules",
    "/etc/motd", "/etc/mysql", "/etc/netplan", "/etc/network/interfaces",
    "/etc/nginx", "/etc/os-release", "/etc/pam.d", "/etc/passwd", "/etc/php",
    "/etc/profile", "/etc/protocols", "/etc/resolv.conf", "/etc/rsyslog.conf",
    "/etc/samba", "/etc/security/limits.conf", "/etc/services", "/etc/shadow",
    "/etc/ssh/ssh_config", "/etc/ssh/sshd_config", "/etc/sudoers",
    "/etc/sysctl.conf", "/etc/sysctl.d", "/etc/systemd/network",
    "/etc/systemd/system", "/etc/timezone", "/etc/udev/rules.d", "/home",
    "/lib", "/lib64", "/media", "/mnt", "/opt", "/opt/homebrew",
    "/opt/homebrew/bin", "/opt/local", "/opt/local/bin", "/private/etc",
    "/private/tmp", "/private/var", "/private/var/log", "/proc/cpuinfo",
    "/proc/filesystems", "/proc/interrupts", "/proc/loadavg", "/proc/meminfo",
    "/proc/modules", "/proc/mounts", "/proc/net", "/proc/partitions",
    "/proc/self", "/proc/stat", "/proc/sys/kernel", "/proc/sys/net/ipv4",
    "/proc/sys/vm", "/proc/uptime", "/proc/version", "/root", "/run/lock",
    "/run/systemd", "/run/user", "/sbin", "/snap/bin", "/srv", "/sys/block",
    "/sys/class", "/sys/class/net", "/sys/class/power_supply", "/sys/devices",
    "/sys/firmware", "/sys/fs/cgroup", "/sys/kernel", "/sys/module", "/tmp",
    "/usr/bin", "/usr/bin/env", "/usr/bin/python3", "/usr/games",
    "/usr/include", "/usr/lib", "/usr/lib/systemd",
    "/usr/lib/x86_64-linux-gnu", "/usr/libexec", "/usr/local",
    "/usr/local/Cellar", "/usr/local/bin", "/usr/local/etc",
    "/usr/local/include", "/usr/local/lib", "/usr/local/opt",
    "/usr/local/sbin", "/usr/local/share", "/usr/sbin", "/usr/share",
    "/usr/share/applications", "/usr/share/doc", "/usr/share/keyrings",
    "/usr/share/man", "/usr/share/zoneinfo", "/var/cache", "/var/lib",
    "/var/lib/apt", "/var/lib/docker", "/var/lib/dpkg", "/var/lib/mysql",
    "/var/lib/postgresql", "/var/lib/systemd", "/var/log", "/var/log/apache2",
    "/var/log/apt", "/var/log/auth.log", "/var/log/dmesg", "/var/log/journal",
    "/var/log/kern.log", "/var/log/messages", "/var/log/nginx",
    "/var/log/syslog", "/var/log/wtmp", "/var/mail", "/var/opt", "/var/run",
    "/var/spool", "/var/tmp", "/var/www", "/var/www/html", "CMakeLists.txt",
    "Cargo.toml", "Dockerfile", "Gemfile", "Vagrantfile", "authorized_keys",
    "build.gradle", "crontab.txt", "docker-compose.yml", "go.mod",
    "id_ed25519.pub", "id_rsa.pub", "known_hosts", "nsswitch.conf",
    "package-lock.json", "package.json", "pom.xml", "pyproject.toml",
    "requirements.txt", "resolv.conf", "setup.py", "sources.list",
    "sshd_config", "~/.aws", "~/.bashrc", "~/.cache", "~/.config",
    "~/.docker", "~/.gitconfig", "~/.kube", "~/.local", "~/.local/bin",
    "~/.npm", "~/.profile", "~/.ssh", "~/.vimrc", "~/.zshrc", "~/Desktop",
    "~/Documents", "~/Downloads", "~/Library", "~/Library/Application",
    "~/Library/Preferences", "~/Movies", "~/Music", "~/Pictures",
)

# Flags, 384 of them: every single-letter form (`-f`, `-c`, ... both cases,
# the most frequent CLI tokens in the corpus by a wide margin, 2 tokens each
# today), the common clusters (`-la`, `-rf`, `-xzf`), find's word flags
# (`-name`, `-type`, `-exec`) and the long `--` forms, ranked by measured
# frequency x tokens saved. The long tail of `--` flags is where this block
# was trimmed to hit 896.

_CLI_FLAG_STRINGS: tuple[str, ...] = (
    "--abort", "--add", "--address", "--after", "--all", "--allow-downgrades",
    "--allow-empty", "--allow-empty-message", "--allow-unauthenticated",
    "--amend", "--apparent-size", "--append", "--archive", "--assume-yes",
    "--author", "--author-date-is-committer-date", "--background", "--backup",
    "--bare", "--binary", "--binary-files", "--bind", "--block-size",
    "--boot", "--branch", "--break-system-packages", "--bwlimit", "--bytes",
    "--cacert", "--cached", "--cert", "--check", "--checksum", "--color",
    "--compress", "--config", "--config-file", "--context", "--continue",
    "--count", "--create", "--csv", "--daemon", "--data", "--data-binary",
    "--data-urlencode", "--date", "--debug", "--delete", "--delete-after",
    "--delete-during", "--delete-excluded", "--depth", "--dereference",
    "--detach", "--devices", "--dir", "--directory", "--disable", "--dns",
    "--domain", "--dry-run", "--edit", "--enable", "--env", "--exclude",
    "--exclude-dir", "--exclude-from", "--extended-regexp",
    "--extra-index-url", "--extract", "--fail", "--field-separator", "--file",
    "--filename", "--files-with-matches", "--files-without-match",
    "--files0-from", "--filter", "--fix-broken", "--fixed-strings",
    "--follow", "--force", "--force-with-lease", "--foreground", "--format",
    "--full", "--get", "--global", "--graph", "--grep", "--group", "--hard",
    "--head", "--header", "--help", "--host", "--hostname", "--http2",
    "--human-numeric-sort", "--human-readable", "--ignore", "--ignore-case",
    "--image", "--include", "--inodes", "--input", "--insecure",
    "--insecure-skip-tls-verify", "--interactive", "--invert-match", "--ipv4",
    "--ipv6", "--iso-8601", "--jobs", "--json", "--keep", "--keep-old-files",
    "--key", "--label", "--legacy-peer-deps", "--limit", "--limit-rate",
    "--line-buffered", "--lines", "--list", "--listen", "--local",
    "--location", "--logfile", "--long", "--match", "--max-count",
    "--max-depth", "--max-line-length", "--max-size", "--max-unchanged-stats",
    "--memory", "--merge", "--mirror", "--mode", "--mount", "--name",
    "--name-only", "--name-status", "--namespace", "--network",
    "--no-cache-filter", "--no-clobber", "--no-color", "--no-dereference",
    "--no-ff", "--no-follow-symlinks", "--no-headers",
    "--no-install-recommends", "--no-optional-locks", "--no-pager",
    "--no-verify", "--no-warn-script-location", "--now", "--null",
    "--numeric-ids", "--numstat", "--offline", "--one-file-system",
    "--oneline", "--only-matching", "--output", "--output-format",
    "--overwrite", "--owner", "--parallel", "--parents", "--partial",
    "--password", "--patch", "--pattern", "--perms", "--pid", "--pidfile",
    "--platform", "--porcelain", "--port", "--prefix", "--priority",
    "--progress", "--protocol", "--proxy", "--prune", "--purge", "--quiet",
    "--range", "--read-only", "--recurse-submodules", "--recursive",
    "--reference", "--regex", "--reinstall", "--release", "--reload",
    "--remote", "--remote-name", "--request", "--retry", "--reverse",
    "--rfc-3339", "--rm", "--rotate", "--rsh", "--rsync-path", "--runtime",
    "--server", "--set", "--short", "--si", "--signoff", "--silent",
    "--simplify-by-decoration", "--since", "--skip", "--socks5", "--sort",
    "--sparse", "--squash", "--staged", "--stat", "--state", "--status",
    "--stdin", "--strict-host-key-checking", "--suffix", "--summary",
    "--system", "--tabs", "--tag", "--tags", "--target", "--target-directory",
    "--template", "--text", "--threads", "--time", "--timeout", "--total",
    "--track", "--transform", "--tty", "--type", "--unique", "--unit",
    "--unix-byte-offsets", "--until", "--update", "--upgrade", "--user",
    "--username", "--utc", "--verbose", "--verify", "--version", "--wait",
    "--watch", "--width", "--with-filename", "--word-regexp", "--zero",
    "--zero-terminated", "-A", "-B", "-C", "-D", "-E", "-F", "-G", "-H", "-I",
    "-J", "-K", "-L", "-M", "-N", "-O", "-P", "-Q", "-R", "-S", "-Ss", "-T",
    "-U", "-V", "-W", "-X", "-Y", "-Z", "-a", "-al", "-and", "-atime", "-av",
    "-b", "-c", "-cf", "-ctime", "-cvf", "-czf", "-czvf", "-d", "-delete",
    "-depth", "-e", "-ef", "-empty", "-eo", "-exec", "-execdir", "-f",
    "-follow", "-g", "-group", "-h", "-i", "-iname", "-iregex", "-it", "-j",
    "-k", "-l", "-la", "-lah", "-lh", "-lname", "-ls", "-lt", "-ltr", "-m",
    "-maxdepth", "-mindepth", "-mmin", "-mtime", "-n", "-name", "-newer",
    "-nogroup", "-not", "-nouser", "-nr", "-o", "-ok", "-or", "-p", "-path",
    "-perm", "-print", "-print0", "-printf", "-prune", "-q", "-ql", "-qq",
    "-quit", "-r", "-regex", "-rf", "-rl", "-rn", "-s", "-sS", "-sf", "-size",
    "-t", "-tf", "-type", "-u", "-user", "-v", "-vv", "-vvv", "-w", "-x",
    "-xdev", "-xf", "-xvf", "-y", "-z",
)

# The three groups are separate only for review and documentation; nothing
# downstream distinguishes them, so they concatenate into one ordered block.
CLI_VOCAB_TOKENS: list[str] = [
    _SPIECE_UNDERLINE + text
    for text in (*_CLI_COMMAND_STRINGS, *_CLI_PATH_STRINGS, *_CLI_FLAG_STRINGS)
]


USER_DEFINED_SYMBOLS: list[str] = (
    EXTRA_ID_TOKENS
    + UL2_MODE_TOKENS
    + STYLE_TOKENS
    + STRUCTURAL_TOKENS
    + RESERVED_TOKENS
    + CLI_TOKENS
    + CLI_VOCAB_TOKENS
)

# Built-in specials assigned via pad_id/eos_id/unk_id, not user_defined_symbols.
# Listed here only so validate_tokenizer.py can check atomicity (Gate 3)
# against the full 1,272-name set without a second source of truth.
BUILTIN_SPECIALS: list[str] = ["<pad>", "</s>", "<unk>"]

ALL_NAMED_SPECIALS: list[str] = BUILTIN_SPECIALS + USER_DEFINED_SYMBOLS


def _validate() -> None:
    assert len(EXTRA_ID_TOKENS) == 256, len(EXTRA_ID_TOKENS)
    assert len(UL2_MODE_TOKENS) == 3, len(UL2_MODE_TOKENS)
    assert len(STYLE_TOKENS) == 12, len(STYLE_TOKENS)
    assert len(STRUCTURAL_TOKENS) == 2, len(STRUCTURAL_TOKENS)
    assert len(RESERVED_TOKENS) == 36, len(RESERVED_TOKENS)
    assert len(CLI_TOKENS) == 64, len(CLI_TOKENS)
    assert len(_CLI_COMMAND_STRINGS) == 219, len(_CLI_COMMAND_STRINGS)
    assert len(_CLI_PATH_STRINGS) == 293, len(_CLI_PATH_STRINGS)
    assert len(_CLI_FLAG_STRINGS) == 384, len(_CLI_FLAG_STRINGS)
    assert len(CLI_VOCAB_TOKENS) == 896, len(CLI_VOCAB_TOKENS)
    assert all(t.startswith(_SPIECE_UNDERLINE) for t in CLI_VOCAB_TOKENS)
    assert len(USER_DEFINED_SYMBOLS) == 1269, len(USER_DEFINED_SYMBOLS)
    assert len(ALL_NAMED_SPECIALS) == 1272, len(ALL_NAMED_SPECIALS)
    assert len(set(ALL_NAMED_SPECIALS)) == len(ALL_NAMED_SPECIALS), "duplicate special token name"


_validate()
