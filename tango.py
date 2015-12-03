import os
import sys
import random

from math import log, exp
from get_data import get_data
from collections import defaultdict
from ngram import BigramLM, DeletedInterpolationTrigrams

start_token = '`'
end_token = '~'

def get_test_set(sents):
    sents = [''.join(sent.split()) for sent in sents]
    return sents

def preprocess_text(data_set):
    return [sent.upper() for sent in data_set]

def populate_vocabulary(training_set):
    vocabulary = defaultdict(int)
    vocabulary[start_token] = len(training_set)
    vocabulary[end_token] = len(training_set)
    for sent in training_set:
        for word in sent:
            vocabulary[word] += 1
    return vocabulary

def main():
    all_sents = get_data()

    training_set = all_sents[:10000]
    test_set = get_test_set(all_sents[10000:])

    training_set = preprocess_text(training_set)
    test_set = preprocess_text(test_set)
    print training_set[3:6]
    print test_set[3:6]

    vocabulary = populate_vocabulary(training_set)

    ###### TRIGRAMS #####
    t_model = TrigramLM(set(vocabulary.keys()))
    t_model.EstimateTrigrams(training_set)
 
    t_model.LaplaceSmooth()
    lambda1, lambda2, lambda3 = DeletedInterpolationTrigrams(held_out_set_prep)
    t_model.Interpolate(lambda1, lambda2, lambda3)

if __name__ == '__main__':
    main()

