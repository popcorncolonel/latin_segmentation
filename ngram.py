import sys
import random
import itertools
import nltk.util

from math import log, exp
from nltk.util import ngrams
from nltk.util import everygrams
from collections import defaultdict
 
class NgramLM:
   def __init__(self, N):
       # maps n -> dict<ngram, count>
       self.model_map = defaultdict(lambda:defaultdict(float))
       self.N = N
       self.max_n = max(N)
       self.log_probs = defaultdict(lambda:float('-inf'))

   def EstimateNgrams(self, training_set):
       for sent in training_set:
           sent = list(sent)
           for ngram in everygrams(sent, max_len=self.max_n):
               n = len(ngram)
               self.model_map[n][ngram] += 1
       for n in self.model_map:
           for ngram in self.model_map[n]:
               if n == 1:
                   pass
               else:
                   self.log_probs[ngram] = log(
                       self.model_map[n][ngram] /
                       self.model_map[n-1][ngram[:-1]]
                   )

class WordDict:
    def __init__(self):
        self.done_token = 'done'
        self.trie = {}

    def populate_words(self, training_set):
        for sent in training_set:
            words = sent.replace('.', '').split()
            for word in words:
                self.insert_word(word)

    def insert_word(self, word):
        history = self.trie
        for c in word:
            if c not in history:
                history[c] = {}
            history = history[c]
        history[self.done_token] = {}

    def is_word(self, word):
        history = self.trie
        for c in word:
            if c not in history:
                return False
            history = history[c]
        return self.done_token in history

    def is_partial_word(self, word_fragment):
        history = self.trie
        for c in word_fragment:
            if c not in history:
                return False
            history = history[c]
        return True


    # returns True if there is some word in the dictionary that ends with char
    def is_ending_char(self, char):
        def helper(history):
            if char in history.keys() and self.done_token in history[char].keys():
                return True
            else:
                for key in history:
                    if helper(history[key]):
                        return True
            return False

        return helper(self.trie)

