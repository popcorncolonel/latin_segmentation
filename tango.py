import os
import sys
import random

from math import log, exp
from get_data import get_data

def get_test_set(sents):
    sents = [''.join(sent.split()) for sent in sents]
    return sents

def preprocess_text(data_set):
    return [sent.upper() for sent in data_set]


def main():
    all_sents = get_data()

    training_set = all_sents[:10000]
    test_set = get_test_set(all_sents[10000:])

    training_set = preprocess_text(training_set)
    test_set = preprocess_text(test_set)
    print training_set[3:6]
    print test_set[3:6]

if __name__ == '__main__':
    main()

