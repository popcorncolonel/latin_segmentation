from get_data import get_data

def get_test_set(sents):
    sents = [''.join(sent.split()) for sent in sents]
    sents = [sent.upper() for sent in sents]
    return ''.join(sents)

def main():
    all_sents = get_data()

    with open('sentences.txt', 'w') as f:
        for sent in all_sents:
            f.write(sent + '\n')

    training_set = all_sents[:10000]
    test_set = get_test_set(all_sents[10000:])
    print test_set

if __name__ == '__main__':
    main()

