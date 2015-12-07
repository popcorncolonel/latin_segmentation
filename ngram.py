import sys
import random
import itertools

from math import log, exp
from nltk.util import ngrams
from collections import defaultdict
 
start_token = '`'  # sentence boundary token.
end_token = '~'  # sentence boundary token.

# for defaultdicts with a nonzero default
def constant_factory(value):
    return itertools.repeat(value).next
 
##################################### Bigrams #########################################
class BigramLM:
    def __init__(self, vocabulary=set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.log_probs = defaultdict(constant_factory(float('-inf')))
 
    def EstimateBigrams(self, training_set):
        for sent in training_set:
            for word in sent:
                self.unigram_counts[word] += 1
        for sent in training_set:
            for i in range(1, len(sent)):
                self.bigram_counts[(sent[i-1], sent[i])] += 1
            # Sentence transitions not accounted for in the above loop.
            self.bigram_counts[(end_token, start_token)] += 1 
        for word1, word2 in self.bigram_counts:
            self.log_probs[(word1, word2)] = log(self.bigram_counts[(word1, word2)] /
                    self.unigram_counts[word1])
 
    def LaplaceSmooth(self):
        new_probs = {}
        length = len(self.vocabulary)
        self.log_probs = defaultdict(constant_factory(log(1.0 / length)))
        for word1, word2 in self.bigram_counts:
            self.log_probs[(word1, word2)] = log((self.bigram_counts[(word1, word2)]+1.0) /
                    (self.unigram_counts[word1]+length))

    def Interpolate(self, lambda1, lambda2):
        length = len(self.vocabulary)

        def unigram_probability(word):
            return self.unigram_counts[word] / length

        def bigram_probability(word1, word2):
            return ((self.bigram_counts[(word1, word2)]+1.0) /
                    (self.unigram_counts[word1]+length))

        self.log_probs = defaultdict(constant_factory(log(1.0 / length)))
        for word1, word2 in self.bigram_counts:
            self.log_probs[(word1, word2)] = log(lambda1 * unigram_probability(word2) + 
                                        lambda2 * bigram_probability(word1, word2))

def DeletedInterpolation(corpus):
    lambda1 = lambda2 = 0
    vocabulary = defaultdict(float)
    bigram_counts = defaultdict(float)

    # Gather data from the corpus
    for sent in corpus:
        for i in range(1, len(sent)):
            # compute unigrams
            vocabulary[sent[i-1]] += 1
            # compute bigrams
            bigram_counts[(sent[i-1], sent[i])] += 1
        vocabulary[end_token] += 1 #not accounted for by loop
    length = len(vocabulary)

    def get_bigram_val(word1, word2):
        uni_val = max(0.0, vocabulary[word1]-1)
        if uni_val <= 0:
            return 0.0
        return max(0.0, bigram_counts[(word1, word2)]-1.0) / uni_val

    def get_unigram_val(word):
        return max(0.0, vocabulary[word]-1.0) / (length-1.0)

    # Actual algorithm
    for (word1, word2) in bigram_counts:
        bigram_val = get_bigram_val(word1, word2)
        unigram_val = get_unigram_val(word1)
        if bigram_val >= unigram_val:
            lambda2 += bigram_counts[(word1, word2)]
        else:
            lambda1 += bigram_counts[(word1, word2)]

    # normalize lambdas
    total = lambda1 + lambda2
    lambda1 /= total
    lambda2 /= total
    return (lambda1, lambda2)
 
############################ Trigrams  ############################

class TrigramLM:
    def __init__(self, vocabulary=set()):
        self.vocabulary = vocabulary
        self.unigram_counts = defaultdict(float)
        self.bigram_counts = defaultdict(float)
        self.trigram_counts = defaultdict(float)
        self.log_probs = defaultdict(constant_factory(float('-inf')))
 
    def EstimateTrigrams(self, training_set):
        for sent in training_set:
            self.unigram_counts[sent[0]] += 1 # <S>
            self.unigram_counts[sent[1]] += 1 # <word>
            self.bigram_counts[(sent[0], sent[1])] += 1 #<S><word>
            self.trigram_counts[(end_token, sent[0], sent[1])] += 1 # </S><S><word>
            for i in range(2, len(sent)):
                self.trigram_counts[(sent[i-2], sent[i-1], sent[i])] += 1
                self.bigram_counts[(sent[i-1], sent[i])] += 1
                self.unigram_counts[sent[i]] += 1
            # Sentence transitions not accounted for in the above loop.
            self.bigram_counts[(end_token, start_token)] += 1  #</S><S>
            self.trigram_counts[(sent[-2], end_token, start_token)] += 1 # <word></S><S>
        for word1, word2, word3 in self.trigram_counts:
            try:
                self.log_probs[(word1, word2, word3)] = log(self.trigram_counts[
                      (word1, word2, word3)] / self.bigram_counts[(word1, word2)])
            except:
                pass
                #print word1, word2, word3
 
    # assigns a nonzero probability to all valid Bigrams
    def LaplaceSmooth(self):
        new_probs = {}
        length = len(self.vocabulary) 
        self.log_probs = defaultdict(constant_factory(log(1.0 / length)))
        for word1, word2, word3 in self.trigram_counts:
            self.log_probs[(word1, word2, word3)] = log((self.trigram_counts[
            (word1, word2, word3)]+1.0) / (self.bigram_counts[(word1, word2)]+length))

    def Interpolate(self, lambda1, lambda2, lambda3):
        length = len(self.vocabulary)

        def unigram_probability(word):
            return self.unigram_counts[word] / length

        def bigram_probability(word1, word2):
            return ((self.bigram_counts[(word1, word2)]+1.0) /
                    (self.unigram_counts[word1]+length))

        def trigram_probability(word1, word2, word3):
            return ((self.trigram_counts[(word1, word2, word3)]+1.0) /
                    (self.bigram_counts[(word1, word2)]+length))

        self.log_probs = defaultdict(constant_factory(log(1.0 / length)))
        for word1, word2, word3 in self.trigram_counts:
            self.log_probs[(word1, word2, word3)] = log(
                                        lambda1 * unigram_probability(word3) + 
                                        lambda2 * bigram_probability(word2, word3) +
                                        lambda3 * trigram_probability(word1, word2, word3))

def DeletedInterpolationTrigrams(corpus):
    lambda1 = lambda2 = lambda3 = 0
    vocabulary = defaultdict(float)
    bigram_counts = defaultdict(float)
    trigram_counts = defaultdict(float)

    # Gather data from the corpus
    for sent in corpus:
        for i in range(2, len(sent)):
            # compute unigrams
            vocabulary[sent[i-2]] += 1
            # compute bigrams
            bigram_counts[(sent[i-2], sent[i-1])] += 1
            # compute trigrams
            trigram_counts[(sent[i-2], sent[i-1], sent[i])] += 1
        vocabulary[sent[-1]] += 1 #not accounted for by loop
        vocabulary[sent[-2]] += 1 #not accounted for by loop
        bigram_counts[(sent[-2], sent[-1])] += 1 #not accounted for by loop
    length = len(vocabulary)

    def get_trigram_val(word1, word2, word3):
        bi_val = bigram_counts[(word1, word2)]-1
        if bi_val <= 0:
            return 0.0
        return max(0.0, trigram_counts[(word1, word2, word3)]-1.0) / bi_val

    def get_bigram_val(word1, word2):
        uni_val = max(0.0, vocabulary[word1]-1)
        if uni_val <= 0:
            return 0.0
        return max(0.0, bigram_counts[(word1, word2)]-1.0) / uni_val

    def get_unigram_val(word):
        return max(0.0, vocabulary[word]-1.0) / (length-1.0)

    # Actual algorithm
    for (word1, word2, word3) in trigram_counts:
        trigram_val = get_trigram_val(word1, word2, word3)
        bigram_val = get_bigram_val(word2, word3)
        unigram_val = get_unigram_val(word3)
        if trigram_val >= bigram_val and trigram_val >= unigram_val:
            lambda3 += trigram_counts[(word1, word2, word3)]
        elif bigram_val > trigram_val and bigram_val > unigram_val:
            lambda2 += trigram_counts[(word1, word2, word3)]
        else:
            lambda1 += trigram_counts[(word1, word2, word3)]

    # normalize lambdas
    total = lambda1 + lambda2 + lambda3
    lambda1 /= total
    lambda2 /= total
    lambda3 /= total
    return (lambda1, lambda2, lambda3)

########################## </Trigrams> ############################

class QuadgramLM:
    def __init__(self):
        self.quadgram_counts = defaultdict(float)
 
    def EstimateQuadgrams(self, training_set):
        for sent in training_set:
            quadgrams = ngrams(list(sent), 4)
            for quad in quadgrams:
                self.quadgram_counts[quad] += 1

class NgramLM:
   def __init__(self, n):
       self.counts = defaultdict(float)
       self.n = n

   def EstimateNgrams(self, training_set):
       for sent in training_set:
           grams = ngrams(list(sent), self.n)
           for ngram in grams:
               self.counts[ngram] += 1

 
