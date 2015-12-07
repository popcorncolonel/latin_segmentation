import os
import sys
import random

from math import log, exp
from get_data import get_data
from collections import defaultdict
from ngram import BigramLM, TrigramLM, DeletedInterpolationTrigrams

start_token = '`'
end_token = '~'

def despace_sents(sents):
    sents = [''.join(sent.split()) for sent in sents]
    return sents

def preprocess_text(data_set):
    return [start_token + sent.upper() + end_token for sent in data_set]

def populate_vocabulary(training_set):
    vocabulary = defaultdict(int)
    vocabulary[start_token] = len(training_set)
    vocabulary[end_token] = len(training_set)
    for sent in training_set:
        for word in sent:
            vocabulary[word] += 1
    return vocabulary

class Segmenter:
    def __init__(self, bigram_model=None,
                 trigram_model=None,
                 quadgram_model=None):
        self.bigram_model = bigram_model
        self.trigram_model = trigram_model
        self.quadgram_model = quadgram_model

    def tango(self, sent):
        return sent

    def ngrams(self, sent, n=2):
        sent = list(sent)
        new_sent = []
        for i, c2 in enumerate(sent):
            if i < n:
                new_sent.append(c2)
                continue
            c1 = sent[i-1]
            space_prob = self.bigram_model.log_probs[(c1, ' ')]
            this_prob = self.bigram_model.log_probs[(c1, c2)]
            if space_prob > 0.675 * this_prob:
                new_sent.append(' ')
            new_sent.append(c2)
        return ''.join(new_sent)

    # Under the assumption that sent.split() == correct.split().
    def evaluate(self, sent, correct):
        n_characters = len(sent.replace(' ', ''))
        sent = list(sent)
        correct = list(correct)
        i = j = 0
        false_positives = 0.0
        false_negatives = 0.0
        n_correct = 0.0
        while True:
            if i == len(sent) or j == len(correct):
                break
            c1 = sent[i]
            c2 = correct[j]
            if c2 == ' ' and c1 == ' ':
                n_correct += 1
                i += 1
                j += 1
            elif c2 == ' ': # we guessed there shouldn't be a space, but there is
                false_negatives += 1
                j += 1 
            elif c1 == ' ': # we guessed space, but it shouldn't be
                false_positives += 1
                i += 1
            else: # c1 == c2 != ' '
                if sent[i-1] != ' ' and correct[j-1] != ' ': # correctly guessed that there shouldn't be a 
                    n_correct += 1
                i += 1
                j += 1
        return n_correct / n_characters

def get_local_data():
    with open('sentences.txt') as f:
        return [s.strip() for s in list(f)]

def main():
    #all_sents = get_data()
    all_sents = get_local_data()

    all_sents = preprocess_text(all_sents)
    training_set = all_sents[:9000]
    held_out_set = all_sents[9000:10000]
    test_set = despace_sents(all_sents[10000:])

    '''
    print training_set[0]
    print
    print held_out_set[0]
    print
    '''
    print all_sents[10000]
    print test_set[0]

    vocabulary = populate_vocabulary(training_set)

    model = BigramLM(set(vocabulary.keys()))
    model.EstimateBigrams(training_set)

    seg = Segmenter(model)
    segmented = seg.ngrams(test_set[0])
    print segmented
    score = seg.evaluate(segmented, all_sents[10000])
    print score

    '''
    t_model = TrigramLM(set(vocabulary.keys()))
    t_model.EstimateTrigrams(training_set)
 
    t_model.LaplaceSmooth()
    lambda1, lambda2, lambda3 = DeletedInterpolationTrigrams(held_out_set)
    print lambda1, lambda2, lambda3
    t_model.Interpolate(lambda1, lambda2, lambda3)
    '''

if __name__ == '__main__':
    main()

