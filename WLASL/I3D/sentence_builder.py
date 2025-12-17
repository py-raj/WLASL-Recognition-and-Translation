# sentence_builder.py 13.12.25

import time


class SentenceBuilder:
    def __init__(self, min_gap=1.0, max_idle_time=3.0):
        """
        min_gap: minimum seconds between same word additions
        max_idle_time: sentence ends if no new word for this time
        """
        self.words = []
        self.last_word = None
        self.last_time = 0
        self.last_update = time.time()

        self.min_gap = min_gap
        self.max_idle_time = max_idle_time

    def add_word(self, word):
        now = time.time()

        # Ignore repeated word too quickly
        if word == self.last_word and (now - self.last_time) < self.min_gap:
            return None

        self.words.append(word)
        self.last_word = word
        self.last_time = now
        self.last_update = now

        return self.get_sentence()

    def is_sentence_complete(self):
        idle_time = time.time() - self.last_update
        return idle_time > self.max_idle_time

    def get_sentence(self):
        if not self.words:
            return ""

        sentence = " ".join(self.words)
        return sentence.capitalize()

    def reset(self):
        self.words = []
        self.last_word = None
        self.last_time = 0
        self.last_update = time.time()
