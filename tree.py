#!/usr/bin/env python3
import csv
import spacy
from itertools import chain
from progressbar import ProgressBar
from spacy.symbols import amod, conj, pobj

INPUT = './crop.csv'
OUTPUT = './donato.csv'
LIMIT = None

nlp = spacy.load('en_core_web_sm')

i = 0
with open(INPUT, 'r') as f, open(OUTPUT, 'w') as w:
    reader = csv.DictReader(f)
    writer = csv.DictWriter(w, fieldnames=reader.fieldnames + ['key'])
    writer.writeheader()

    bar = ProgressBar()

    for line in bar(reader):
        i += 1

        text = line['text']
        doc = nlp(text)

        # print('"%s":' % text)

        sentence = list(doc.sents)[0]
        root = sentence.root

        stack = [(root, 0)]
        done = set()
        key = []

        while stack:
            token, lvl = stack.pop()

            if token in done:
                continue

            done.add(token)

            if lvl == 0:
                # print('[%s]' % token.lemma_)
                key.append((lvl, 0, token.lemma_))

            elif (
                token.dep == amod or
                token.dep == conj or
                token.dep_ == 'compound'
            ):
                # print(('--' * lvl) + token.lemma_)
                key.append((lvl, 1, token.lemma_))

            elif token.dep == pobj:
                # print(('--' * lvl) + '[%s]' % token.lemma_)
                key.append((lvl, 2, token.lemma_))

            for child in token.subtree:
                stack.append((child, lvl + 1))

        key = ' '.join([i[2] for i in sorted(key)])
        # print('Key: "%s"' % key)

        row = line.copy()
        row['key'] = key
        writer.writerow(row)

        # print()

        if LIMIT and i == LIMIT:
            break
