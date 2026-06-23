import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_probs = self.analyze_bigrams(corpus)

    def simple_tokenizer(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def analyze_bigrams(self, corpus):
        if isinstance(corpus, list):
            text = " ".join(corpus)
        else:
            text = corpus

        words = self.simple_tokenizer(text)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        bigram_probs = defaultdict(dict)

        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]

        return bigram_probs

    def generate_text(self, start_word, length=20):
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)

            if not next_words:
                break

            next_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values()),
            )[0]

            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)
