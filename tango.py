import os
import sys
import time
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
        #sent = start_token + sent + end_token
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


def I_gt(y, z):
    if y > z:
        return 1
    return 0

# Vote for whether there should be a space at location k
def tango_vote(segmenter, sent, k, n):
    left = sent[k-n : k]
    right = sent[k : k+n]
    total = 0.0
    for d in [left, right]:
        for j in range(1, n):
            straddling = sent[k-j : k-j+n] 
            s_d = segmenter.model_map[n][tuple(d)]
            tj = segmenter.model_map[n][tuple(straddling)]
            '''
            # debugging
            if k == 15 and n == 5:
                print tuple(d), '>', tuple(straddling)
                print s_d, '>', tj
                print I_gt(s_d, tj)
                print
            '''
            total += I_gt(s_d, tj)
    return total / (2 * (n-1))

def ngram_vote(segmenter, sent, k, n):
    ''' If "A B" is more likely than "AB", return 1.'''
    sent = list(sent)
    left = sent[k-n/2 : k]
    right = sent[k : k+n/2]
    space_prob = segmenter.log_probs[tuple(left)+tuple(' ')+tuple(right)]
    this_prob = segmenter.log_probs[tuple(left)+tuple(right)]
    return I_gt(space_prob, this_prob)

'''
def ngrams(self, sent, n=6):
    # If "A B" is more likely than "AB", insert a space.
    sent = list(sent)
    new_sent = []
    orig_n = n
    for i, c2 in enumerate(sent):
        n = orig_n
        while i < n/2 or i > len(sent)-n/2:
            n -= 2
        if n <= 0:
            new_sent.append(c2)
            continue
        left = sent[i-n/2 : i]
        right = sent[i : i+n/2]
        space_prob = self.log_probs[tuple(left)+tuple(' ')+tuple(right)]
        this_prob = self.log_probs[tuple(left)+tuple(right)]
        if space_prob > this_prob:
            new_sent.append(' ')
        new_sent.append(c2)
    return ''.join(new_sent)
'''

# model_map is a map from ints to NgramLM's
#           N -> dict<Ngram, count>
class Segmenter:
    # model_map is unsupervised
    # log_probs is supervised
    def __init__(self, model_map, log_probs, N):
        self.model_map = model_map
        self.log_probs = log_probs
        self.N = N

    def tango(self, sent, vote_function=tango_vote):
        # average all possible n values
        def total_v(k, N):
            total = 0.0
            length = len(N)
            for n in N:
                if k-n < 0: # if the n-gram doesn't exist to the left
                    length -= 1
                    continue
                if k+n > len(sent): # if the n-gram doesn't exist to the right
                    length -= 1
                    continue
                total += vote_function(self, sent, k, n)
            if length == 0:
                return 0
            return total / length

        votes = []

        for k in range(len(sent)):
            votes.append(total_v(k, self.N))

        t = 1.0

        '''
        # Debugging
        for i, v in enumerate(votes):
            print sent[i], v
        '''

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
                if should_insert_space(l):
                    new_sent.append(' ')
                new_sent.append(sent[l])
            return ''.join(new_sent)

        return insert_spaces(sent)

    # Precision = tp / (tp + fp)
    def precision(self, parsed, correct):
        n_characters = len(parsed.replace(' ', ''))
        parsed = list(parsed)
        correct = list(correct)
        i = j = 0
        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0
        true_negatives = 0.0
        while True:
            if i == len(parsed) or j == len(correct):
                break
            c1 = parsed[i]
            c2 = correct[j]
            if c2 == ' ' and c1 == ' ':
                true_positives += 1
                i += 1
                j += 1
            elif c2 == ' ': # we guessed there shouldn't be a space, but there is
                false_negatives += 1
                j += 1 
            elif c1 == ' ': # we guessed space, but it shouldn't be
                false_positives += 1
                i += 1
            else: # c1 == c2 != ' '
                if parsed[i-1] != ' ' and correct[j-1] != ' ': # correctly guessed that there shouldn't be a 
                    true_negatives += 1
                i += 1
                j += 1
        if true_positives == 0:
            return 0
        return true_positives / (true_positives + false_positives)

    # Recall = tp / (tp + fn)
    def recall(self, parsed, correct):
        n_characters = len(parsed.replace(' ', ''))
        parsed = list(parsed)
        correct = list(correct)
        i = j = 0
        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0
        true_negatives = 0.0
        while True:
            if i == len(parsed) or j == len(correct):
                break
            c1 = parsed[i]
            c2 = correct[j]
            if c2 == ' ' and c1 == ' ':
                true_positives += 1
                i += 1
                j += 1
            elif c2 == ' ': # we guessed there shouldn't be a space, but there is
                false_negatives += 1
                j += 1 
            elif c1 == ' ': # we guessed space, but it shouldn't be
                false_positives += 1
                i += 1
            else: # c1 == c2 != ' '
                if parsed[i-1] != ' ' and correct[j-1] != ' ': # correctly guessed that there shouldn't be a 
                    true_negatives += 1
                i += 1
                j += 1
        if true_positives == 0:
            return 0
        return true_positives / (true_positives + false_negatives)

    def F(self, parsed, correct):
        p = self.precision(parsed, correct)
        r = self.recall(parsed, correct)
        if p == r == 0.0:
            return 0.0
        return 2 * (p * r) / (p+r)


    def baseline(self, sent):
        ''' Put spaces everywhere '''
        middle = sent[1:-2]
        return sent[:1] + middle.replace('', ' ') + sent[-2:]

    def eval_test_set(self, test_set, method):
        total_F = 0.0
        total_P = 0.0
        total_R = 0.0
        length = len(test_set)
        for correct in test_set:
            if correct.count(' ') == 0.0:
                length -= 1
                continue
            sent = self.tango(despace(correct), vote_function=method)
            F = self.F(sent, correct)
            if F == 0.0:
                length -= 1
                continue
            '''
            if F < 0.3:
                print F
                print 'Correct:', correct
                print 'Guessed:', sent
                print
                time.sleep(0.5)
            '''

            total_F += F
            total_P += self.precision(sent, correct)
            total_R += self.recall(sent, correct)
        print 'Precision:', total_P / length
        print 'Recall:', total_R / length
        return total_F / length

def get_local_data():
    with open('sentences.txt') as f:
        return [s.strip() for s in list(f)]

def main():
    #all_sents = get_data()
    all_sents = get_local_data()

    all_sents = preprocess_text(all_sents)
    k = 11000
    training_set = all_sents[:k]
    test_set = all_sents[k:]

    N = [7, 4, 5, 6]

    # TODO: max valid word metric
    word_dict = ngram.WordDict()
    word_dict.populate_words(training_set)

    supervised_model = ngram.NgramLM(N)
    model = ngram.NgramLM(N)
    print 'Learning n-gram model...'
    model.EstimateNgrams(despace_sents(training_set))
    supervised_model.EstimateNgrams(training_set)
    print 'Learned n-gram model.'

    seg = Segmenter(model.model_map,
                    supervised_model.log_probs,
                    N)

    #for k, v in sorted(model.model_map[5].items(), key=lambda x:-1*x[1])[:5]:
    #    print k, 'was seen', int(v), 'times.'

    sent = all_sents[11030]
    #sent = all_sents[11029]

    '''
    segmented = seg.ngrams(despace(sent))
    segmented = seg.tango(despace(sent))
    '''
    for f in [tango_vote, ngram_vote]:
        segmented = seg.tango(despace(sent), vote_function=f)

        print sent
        print despace(sent)
        print segmented

        score = seg.precision(segmented, sent)
        print 'Precision:', score
        score = seg.recall(segmented, sent)
        print 'Recall:', score
        score = seg.F(segmented, sent)
        print 'F-measure:', score

        print
        print '---------------------------------------------'
        print

    #print 'Avg F-measure (baseline): %f' % seg.eval_test_set(test_set, seg.baseline)
    #print
    print 'Avg F-measure (ngrams): %f' % seg.eval_test_set(test_set, ngram_vote)
    print
    print 'Avg F-measure (tango): %f' % seg.eval_test_set(test_set, tango_vote)

if __name__ == '__main__':
    main()

