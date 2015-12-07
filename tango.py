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

class Segmenter:
    def __init__(self, model_map):
        self.model_map = model_map
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
                    s_d = self.model_map[n].counts[tuple(d)]
                    tj = self.model_map[n].counts[tuple(straddling)]
                    total += I_gt(s_d, tj)
            return total / (2 * (n-1))

        # average all possible n values
        def total_v(k, N):
            total = 0.0
            for n in N:
                total += v(k, n)
            return total / len(N)

        votes = []
        N = set([2,3,4])
        for k in range(len(sent)):
            votes.append(total_v(k, N))

        t = 0.95
        print votes

        def should_insert_space(l):
            if l > 0 and l < len(sent)-1:
                # local max => insert space
                if votes[l] > votes[l-1] and votes[l] > votes[l+1]:
                    return True
                if votes[l] > t:
                    return True
            return False

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
    sent = test_set[3]
    print sent
    print despace(sent)

    vocabulary = populate_vocabulary(training_set)
    training_set = despace_sents(training_set)

    b_model = ngram.NgramLM(2)
    b_model.EstimateNgrams(training_set)

    t_model = ngram.NgramLM(3)
    t_model.EstimateNgrams(training_set)

    q_model = ngram.NgramLM(4)
    q_model.EstimateNgrams(training_set)

    model_map = {
            2: b_model,
            3: t_model,
            4: q_model
    }

    seg = Segmenter(model_map)

    #segmented = seg.ngrams(despace(test_set[0]))
    segmented = seg.tango(despace(sent))
    print segmented
    score = seg.evaluate(segmented, sent)
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

