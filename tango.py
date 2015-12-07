import os
import sys
import ngram
import random

from math import log, exp
from get_data import get_data
from collections import defaultdict
#from ngram import BigramLM, TrigramLM

start_token = '`'
end_token = '~'

def despace(sent):
    return ''.join(sent.split())

def despace_sents(sents):
    return [despace(sent) for sent in sents]

def preprocess_text(data_set):
    new_data_set = []
    for sent in data_set:
        sent = sent.upper()
        sent = sent.replace('U', 'V')
        sent = sent.replace('J', 'I')
        sent = start_token + sent + end_token
        new_data_set.append(sent)
    return new_data_set

def populate_vocabulary(training_set):
    vocabulary = defaultdict(int)
    vocabulary[start_token] = len(training_set)
    vocabulary[end_token] = len(training_set)
    for sent in training_set:
        for word in sent:
            vocabulary[word] += 1
    return vocabulary

# model_map is a map from ints to NgramLM's
#           N -> dict<Ngram, count>
class Segmenter:
    def __init__(self, model_map, N):
        self.model_map = model_map
        self.N = N

    def tango(self, sent):
        def I_gt(y, z):
            if y > z:
                return 1
            return 0

        # vote for whether there should be a space at location k
        def v(k, n):
            if k-n < 0: # if the n-gram doesn't exist to the left
                return 0
            if k+n > len(sent): # if the n-gram doesn't exist to the right
                return 0

            left = sent[k-n : k]
            right = sent[k : k+n]
            total = 0.0
            for d in [left, right]:
                for j in range(1, n):
                    straddling = sent[k-j : k-j+n] 
                    s_d = self.model_map[n][tuple(d)]
                    tj = self.model_map[n][tuple(straddling)]
                    total += I_gt(s_d, tj)
            return total / (2 * (n-1))

        # average all possible n values
        def total_v(k, N):
            total = 0.0
            for n in N:
                total += v(k, n)
            return total / len(N)

        votes = []
        for k in range(len(sent)):
            votes.append(total_v(k, self.N))

        t = 0.95

        def should_insert_space(l):
            if l > 0 and l < len(sent)-1:
                # local max => insert space
                if votes[l] > votes[l-1] and votes[l] > votes[l+1]:
                    return True
                if votes[l] > t:
                    return True
            return False

        # TODO: are we putting spaces in the right place?
        def insert_spaces(sent):
            new_sent = []
            for l in range(len(sent)):
                new_sent.append(sent[l])
                if should_insert_space(l):
                    new_sent.append(' ')
            return ''.join(new_sent)

        return insert_spaces(sent)

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
    test_set = all_sents[10000:]

    '''
    print training_set[0]
    print
    print held_out_set[0]
    print
    '''
    sent = random.choice(test_set)
    print sent
    print despace(sent)

    vocabulary = populate_vocabulary(training_set)
    training_set = despace_sents(training_set)

    N = set([4,5,6,7,8])
    print 'Learning n-gram model...'
    model = ngram.NgramLM(N)
    model.EstimateNgrams(training_set)
    print 'Learned n-gram model.'

    for k, v in sorted(model.model_map[4].items(), key=lambda x:-1*x[1])[:5]:
        print k, 'was seen', int(v), 'times.'

    seg = Segmenter(model.model_map, N)

    #segmented = seg.ngrams(despace(test_set[0]))
    segmented = seg.tango(despace(sent))
    print segmented
    score = seg.evaluate(segmented, sent)
    print score

if __name__ == '__main__':
    main()

