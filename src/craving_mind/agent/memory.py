"""Agent memory: persistent workspace files (bible.md, graveyard.md, compress.py)."""

import os
import re


# Maximum epitaphs kept in graveyard. Oldest are dropped when limit exceeded.
_MAX_GRAVEYARD_ENTRIES = 5


class MemoryManager:
    """Manages agent's persistent workspace files."""

    # Minimal stub — shows the signature and basic constraint.
    # Agent must figure out what the function does from the scores alone.
    _SEED_COMPRESS = '''\
def compress(s: str, target_ratio: float) -> str:
    """Transform input. Output length must be <= target_ratio * len(s)."""
    target_len = max(1, int(len(s) * target_ratio))
    return s[:target_len]
'''

    def __init__(self, config: dict, agent_dir: str):
        self.config = config
        self.agent_dir = agent_dir
        self.bible_max_weight_pct = config["memory"]["bible_max_weight_pct"]
        os.makedirs(agent_dir, exist_ok=True)
        self._seed_compress()

    def _seed_compress(self):
        """Write seed compress.py if none exists yet."""
        path = os.path.join(self.agent_dir, "compress.py")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._SEED_COMPRESS)

    def read_file(self, filename: str) -> str:
        path = os.path.join(self.agent_dir, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def write_file(self, filename: str, content: str):
        path = os.path.join(self.agent_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def backup(self) -> dict:
        """Backup current state (for OOM rollback)."""
        return {f: self.read_file(f) for f in ["bible.md", "graveyard.md", "compress.py"]}

    def restore(self, backup: dict):
        """Restore from backup (OOM rollback)."""
        for filename, content in backup.items():
            if content:
                self.write_file(filename, content)

    def bible_token_weight(self, token_counter) -> float:
        """bible.md size as fraction of a budget (estimated tokens)."""
        bible = self.read_file("bible.md")
        return token_counter.estimate(bible)

    def init_from_inheritance(self, prev_compress: str = None, prev_graveyard: str = None):
        """Initialize from predecessor's artifacts."""
        if prev_compress:
            self.write_file("compress.py", prev_compress)
        if prev_graveyard:
            self.write_file("graveyard.md", prev_graveyard)

    # ------------------------------------------------------------------
    # Graveyard: compact format, one line per entry
    # ------------------------------------------------------------------
    # Format: "E<epoch> <pass>/<total> best:a=<x>,b=<y> | <last_words>"
    # Example: "E3 0/10 best:a=0.88,b=0.60 | TF-IDF extraction peaked at a=0.88 but b never crossed 0.60"

    @staticmethod
    def parse_graveyard(content: str) -> list[str]:
        """Parse graveyard into list of entry lines."""
        if not content or not content.strip():
            return []
        return [line for line in content.strip().split("\n") if line.strip()]

    def append_epitaph(self, entry_line: str):
        """Append one epitaph line and trim to max entries."""
        entries = self.parse_graveyard(self.read_file("graveyard.md"))
        entries.append(entry_line.strip())
        # Keep only the most recent entries.
        entries = entries[-_MAX_GRAVEYARD_ENTRIES:]
        self.write_file("graveyard.md", "\n".join(entries) + "\n")
