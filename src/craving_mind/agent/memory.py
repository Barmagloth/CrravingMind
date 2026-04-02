"""Agent memory: persistent workspace files (bible.md, graveyard.md, compress.py)."""

import os
import re


class MemoryManager:
    """Manages agent's persistent workspace files."""

    # Minimal working compress.py that the agent can improve.
    # Extractive: scores sentences by position + keyword density, picks top-N.
    _SEED_COMPRESS = '''\
import re

def compress(text: str, target_ratio: float) -> str:
    """Compress text to approximately target_ratio of original length."""
    if not text or target_ratio <= 0:
        return ""
    target_len = max(1, int(len(text) * target_ratio))
    # Normalise whitespace
    text = re.sub(r"\\s+", " ", text.strip())
    if len(text) <= target_len:
        return text
    # Split into sentences
    sentences = re.split(r"(?<=[.!?;])\\s+", text)
    if not sentences:
        return text[:target_len]
    # Score each sentence: early position + length bonus + capital-word density
    scored = []
    for i, s in enumerate(sentences):
        words = s.split()
        if not words:
            continue
        caps = sum(1 for w in words if w[0].isupper()) / len(words)
        pos = 1.0 - i / max(len(sentences), 1)
        score = 0.4 * pos + 0.3 * caps + 0.3 * min(len(s) / 120, 1.0)
        scored.append((score, i, s))
    scored.sort(reverse=True)
    # Greedily pick top sentences in original order until target_len
    picked_indices = set()
    total = 0
    for _score, idx, s in scored:
        if total + len(s) + 1 > target_len:
            continue
        picked_indices.add(idx)
        total += len(s) + 1
    if not picked_indices:
        picked_indices.add(scored[0][1])
    result = " ".join(s for _sc, idx, s in sorted(scored, key=lambda x: x[1]) if idx in picked_indices)
    return result[:target_len]
'''

    def __init__(self, config: dict, agent_dir: str):
        self.config = config
        self.agent_dir = agent_dir
        self.graveyard_ttl = config["memory"]["graveyard_ttl_epochs"]
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
            tagged = prev_graveyard.replace(
                "<!-- AMENDMENT:", "<!-- AMENDMENT:inherited "
            )
            self.write_file("graveyard.md", tagged)

    def cleanup_graveyard(self, current_epoch: int):
        """Remove expired entries from graveyard (TTL-based)."""
        content = self.read_file("graveyard.md")
        if not content:
            return

        # Parse amendment blocks: <!-- AMENDMENT:epoch=N --> ... <!-- /AMENDMENT -->
        pattern = re.compile(
            r"<!--\s*AMENDMENT[^>]*epoch=(\d+)[^>]*-->.*?<!--\s*/AMENDMENT\s*-->",
            re.DOTALL,
        )

        def keep_block(m):
            epoch_written = int(m.group(1))
            if current_epoch - epoch_written > self.graveyard_ttl:
                return ""
            return m.group(0)

        cleaned = pattern.sub(keep_block, content)
        self.write_file("graveyard.md", cleaned)
