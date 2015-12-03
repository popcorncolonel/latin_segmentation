import os
import re
import sys
import random
from BeautifulSoup import BeautifulSoup

data_files = [f for f in os.listdir('data') if f.endswith('.html')]

def search_data_files(string):
    possible_files = []
    for d in data_files:
        if string in open('data/'+d).read():
            possible_files.append(d)
    print possible_files
    sys.exit()

char_match = re.compile('[a-zA-Z\. ]', flags=re.I)

def get_sents(text):
    sents = [sent.strip() for sent in text.split('.')]
    return [sent+'.' for sent in sents if len(sent) > 0]

def clean(text):
    new_text = []
    for c in text:
        if re.match(char_match, c):
            if c == ' ':
                if len(new_text) == 0 or new_text[-1] != ' ':
                    new_text += c
            else:
                new_text += c
                if c == '.':
                    new_text += ' '
        else:
            if len(new_text) == 0 or new_text[-1] != ' ':
                new_text += ' '
    return ''.join(new_text)

def get_text(filename):
    with open(filename) as f:
        html_doc = f.read()
        soup = BeautifulSoup(html_doc)
        all_text = ''.join(soup.findAll(text=True))
        all_text = clean(all_text)
    return all_text


def get_data():
    sents = []
    for filename in data_files:
        text = get_text('data/'+filename)
        sents.extend(get_sents(text))
    return sents

