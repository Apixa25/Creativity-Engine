"""
Git Monitor — Watches for new commits and feeds them to the engine.

Polls the git repo at a configurable interval. When a new commit is detected,
it extracts the commit message, diff, and file stats, then passes them to
the engine for direct review — NOT through the creative association pipeline.

This is the "friend reviewing your code" path, not the "friend going on tangents" path.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CommitInfo:
    """Everything the engine needs to give thoughtful feedback on a commit."""
    hash: str
    hash_short: str
    message: str
    author: str
    timestamp: str
    files_changed: list[str] = field(default_factory=list)
    stats: str = ""             # e.g. "3 files changed, 42 insertions(+), 7 deletions(-)"
    diff: str = ""              # truncated diff content
    branch: str = ""

    @property
    def summary_line(self) -> str:
        return f"{self.hash_short} — {self.message.splitlines()[0][:80]}"


class GitMonitor:
    """Polls the local git repo for new commits.

    Designed to run as an async loop alongside the heartbeat.
    When a new commit appears, it fires a callback with the full CommitInfo.
    """

    def __init__(
        self,
        repo_path: str | Path = ".",
        poll_interval_seconds: int = 30,
        max_diff_chars: int = 4000,
        watch_branch: str | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.poll_interval = poll_interval_seconds
        self.max_diff_chars = max_diff_chars
        self.watch_branch = watch_branch
        self._last_hash: str | None = None
        self._running = False
        self._available = False

    def initialize(self) -> bool:
        """Verify this is a git repo and store the current HEAD."""
        try:
            head = self._git("rev-parse", "HEAD")
            if head:
                self._last_hash = head.strip()
                branch = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()
                self._available = True
                print(f"   [Git] Monitor: active on branch '{branch}'")
                print(f"   [Git] Current HEAD: {self._last_hash[:8]}")
                print(f"   [Git] Polling every {self.poll_interval}s for new commits")
                return True
        except Exception as e:
            print(f"   [Git] Monitor: not available — {e}")

        self._available = False
        return False

    @property
    def is_available(self) -> bool:
        return self._available

    async def start(self, on_commit) -> None:
        """Start the polling loop. Calls on_commit(CommitInfo) when a new commit is detected."""
        if not self._available:
            return

        self._running = True
        while self._running:
            await asyncio.sleep(self.poll_interval)
            if not self._running:
                break

            try:
                current_head = await self._git_async("rev-parse", "HEAD")
                current_head = current_head.strip()

                if current_head != self._last_hash:
                    commit_info = await self._extract_commit(current_head)
                    old_short = self._last_hash[:8] if self._last_hash else "none"
                    self._last_hash = current_head
                    print(f"\n   [Git] 🆕 New commit detected! {old_short} → {commit_info.hash_short}")
                    await on_commit(commit_info)

            except Exception as e:
                print(f"   [Git] Poll error: {e}")

    def stop(self) -> None:
        self._running = False

    async def _extract_commit(self, commit_hash: str) -> CommitInfo:
        """Pull all useful information from a commit."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._extract_commit_sync, commit_hash)

    def _extract_commit_sync(self, commit_hash: str) -> CommitInfo:
        """Synchronous commit extraction — runs in executor."""
        hash_short = self._git("rev-parse", "--short", commit_hash).strip()

        log_format = "%H%n%h%n%s%n%an%n%ai%n%B"
        log_output = self._git("log", "-1", f"--format={log_format}", commit_hash)
        lines = log_output.strip().split("\n")

        full_hash = lines[0] if len(lines) > 0 else commit_hash
        short_hash = lines[1] if len(lines) > 1 else hash_short
        subject = lines[2] if len(lines) > 2 else ""
        author = lines[3] if len(lines) > 3 else ""
        timestamp = lines[4] if len(lines) > 4 else ""
        full_message = "\n".join(lines[5:]).strip() if len(lines) > 5 else subject

        files_raw = self._git("diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash)
        files_changed = [line.strip() for line in files_raw.strip().split("\n") if line.strip()]

        stat = self._git("diff", "--stat", f"{commit_hash}~1", commit_hash)
        stat_summary = stat.strip().split("\n")[-1].strip() if stat.strip() else ""

        diff = self._git("diff", f"{commit_hash}~1", commit_hash)
        if len(diff) > self.max_diff_chars:
            diff = diff[:self.max_diff_chars] + f"\n\n... [diff truncated at {self.max_diff_chars} chars] ..."

        branch = self._git("rev-parse", "--abbrev-ref", "HEAD").strip()

        return CommitInfo(
            hash=full_hash,
            hash_short=short_hash,
            message=full_message,
            author=author,
            timestamp=timestamp,
            files_changed=files_changed,
            stats=stat_summary,
            diff=diff,
            branch=branch,
        )

    def _git(self, *args: str) -> str:
        """Run a git command synchronously and return stdout."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 and "unknown revision" not in result.stderr:
            pass
        return result.stdout

    async def _git_async(self, *args: str) -> str:
        """Run a git command asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._git, *args)

    async def check_now(self) -> CommitInfo | None:
        """Force an immediate check — returns CommitInfo if there's a new commit, else None."""
        if not self._available:
            return None

        current_head = await self._git_async("rev-parse", "HEAD")
        current_head = current_head.strip()

        if current_head != self._last_hash:
            commit_info = await self._extract_commit(current_head)
            self._last_hash = current_head
            return commit_info

        return None
