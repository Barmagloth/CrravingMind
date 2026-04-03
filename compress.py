import re
import math
from collections import Counter

def compress(s: str, target_ratio: float) -> str:
    """Extractive summarization - hybrid approach: consecutive start + important sentences."""

    # Handle edge cases
    if not s or not isinstance(s, str):
        return ""

    target_len = max(1, int(len(s) * target_ratio))

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', s.strip())
    sentences = [sent.strip() for sent in sentences if sent.strip()]

    if not sentences:
        return s[:target_len] if len(s) > target_len else s

    if len(sentences) == 1:
        sent = sentences[0]
        return sent if len(sent) <= target_len else sent[:target_len]

    # Calculate word frequencies for importance scoring
    all_words = re.findall(r'\b\w+\b', s.lower())
    word_freq = Counter(all_words)

    # Define stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'as', 'if'
    }

    def score_importance(sentence):
        """Score importance based on content words and entity presence."""
        words = re.findall(r'\b\w+\b', sentence.lower())
        content_words = [w for w in words if w not in stopwords]

        # Score based on rare/important content words
        if not content_words:
            return 0.0

        # TF-IDF style: rare words score higher
        content_score = sum(1.0 / (1.0 + math.log(word_freq.get(w, 1) + 1))
                           for w in content_words)

        # Add entity bonus: proper nouns, numbers, all-caps
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+', sentence))
        numbers = len(re.findall(r'\b\d+', sentence))
        allcaps = len(re.findall(r'\b[A-Z]{2,}\b', sentence))
        entity_score = (proper_nouns + numbers + allcaps) * 1.5

        return content_score + entity_score

    # Always include first sentence for context
    selected = {0}
    current_length = len(sentences[0]) + 1

    # Greedily add consecutive sentences while possible
    for i in range(1, len(sentences)):
        sent = sentences[i]
        sent_length = len(sent) + 1
        if current_length + sent_length <= target_len:
            selected.add(i)
            current_length += sent_length
        else:
            break

    # If we still have space, add high-scoring sentences from later in text
    if current_length < target_len and len(selected) < len(sentences):
        remaining = [(i, score_importance(sentences[i]))
                    for i in range(len(sentences)) if i not in selected]
        remaining.sort(key=lambda x: -x[1])

        for sent_idx, score in remaining:
            if score <= 0:
                continue
            sent = sentences[sent_idx]
            sent_length = len(sent) + 1
            if current_length + sent_length <= target_len:
                selected.add(sent_idx)
                current_length += sent_length

    # Build result in original sentence order
    result_sentences = [sentences[i] for i in sorted(selected)]
    result = ' '.join(result_sentences)

    # Ensure we don't exceed target length
    if len(result) > target_len:
        result = result[:target_len]
        last_space = result.rfind(' ')
        if last_space > target_len // 2:
            result = result[:last_space]

    return result
