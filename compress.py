import re
import math
from collections import Counter

def compress(s: str, target_ratio: float) -> str:
    """Extractive summarization with entity preservation and coherence scoring."""

    # Handle edge cases
    if not s or not isinstance(s, str):
        return ""

    target_len = max(1, int(len(s) * target_ratio))

    # Split into sentences, preserve newlines in split
    sentences = re.split(r'(?<=[.!?])\s+', s.strip())
    sentences = [sent.strip() for sent in sentences if sent.strip()]

    if not sentences:
        return s[:target_len] if len(s) > target_len else s

    if len(sentences) == 1:
        sent = sentences[0]
        return sent if len(sent) <= target_len else sent[:target_len]

    # Calculate word frequencies
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

    def score_sentence(idx, sentence, selected_words=None):
        """Score a sentence based on content, position, and coherence."""
        words = re.findall(r'\b\w+\b', sentence.lower())

        if not words:
            return 0

        # Entity count: proper nouns and numbers
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]+', sentence))
        numbers = len(re.findall(r'\b\d+', sentence))
        entities = proper_nouns + numbers

        # Content score: sum of content word frequencies
        content_words = [w for w in words if w not in stopwords]

        # TF-IDF style scoring: rarer, more content-full words are more valuable
        content_score = sum(1.0 / (1.0 + math.log(word_freq.get(w, 1) + 1))
                           for w in content_words) if content_words else 0

        # Position bonus: favor early sentences
        position = 1.0 + 0.15 * (1.0 - idx / len(sentences)) if len(sentences) > 1 else 1.0

        # Coherence: reward words that appeared in already-selected sentences
        coherence = 0.0
        if selected_words:
            overlap_words = set(content_words) & selected_words
            coherence = len(overlap_words) / (len(content_words) + 1.0) if content_words else 0

        # Combined score
        total_score = (entities * 2.5 + content_score) * position * (1.0 + coherence * 0.5)

        return total_score

    # Always include first sentence
    selected = {0}
    current_length = len(sentences[0]) + 1  # +1 for space after period

    # Track words in selected sentences for coherence scoring
    selected_words = set(re.findall(r'\b\w+\b', sentences[0].lower()))

    # Score and select sentences
    remaining = [(i, s) for i, s in enumerate(sentences) if i != 0]

    # Sort remaining sentences by score (descending), using coherence
    remaining_scored = [(i, score_sentence(i, s, selected_words)) for i, s in remaining]
    remaining_scored.sort(key=lambda x: -x[1])

    # Greedily add sentences with coherence boost
    for sent_idx, score in remaining_scored:
        if score <= 0:
            continue

        sent = sentences[sent_idx]
        sent_length = len(sent) + 1

        if current_length + sent_length <= target_len:
            selected.add(sent_idx)
            current_length += sent_length
            # Add this sentence's content words to selected_words for next iteration
            new_words = [w for w in re.findall(r'\b\w+\b', sent.lower())
                        if w not in stopwords]
            selected_words.update(new_words)

    # Build result in original sentence order
    result_sentences = [sentences[i] for i in sorted(selected)]
    result = ' '.join(result_sentences)

    # Ensure we don't exceed target length
    if len(result) > target_len:
        result = result[:target_len]
        # Find last space to avoid cutting words
        last_space = result.rfind(' ')
        if last_space > target_len // 2:  # Only trim if we can cut off meaningful amount
            result = result[:last_space]

    return result
