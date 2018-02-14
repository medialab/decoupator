#!/usr/bin/env python3
# =============================================================================
# Le Decoupator
# =============================================================================
#
# Magnifique moulinette pour il signore Donato Ricci.
#
import csv
import os
import sys
import json
import numpy as np
from progressbar import ProgressBar
from collections import Counter, defaultdict
from termcolor import colored
from pathlib import Path
from PIL import Image
from helpers import extract_signature, tokenize, weighted_sample, Trie

# =============================================================================
# Variables
# =============================================================================

# Should we sample the results?
SAMPLING = True

# Sample size
SAMPLE_SIZE = 100

# Where are the images located?
IMAGE_FOLDER = 'IMG_EXTREMITIES'

# Where is the metadata JSON file?
METADATA_PATH = 'full_all.json'

# Tweets file
TWEETS_PATH = 'tweets_from_img_list.csv'

# Where should we write the fragments?
OUTPUT_FOLDER = 'output'

# What's our confidence threshold?
CONFIDENCE_THRESHOLD = 1

# What's our percentile threshold
PERCENTILE_THRESHOLD = 75

# What's the minimum prefix length we're gonna use to match captions?
SIGNATURE_THRESHOLD = 2

# Should we reverse the prefix order for caption matchin?
DESC_TFIDF = False

# Should we log the produced clusters
LOG_CLUSTERS = False

# Offset & limit for debugging
OFFSET = None
LIMIT = None

# Whitelist for images, useful for debugging
IMAGES_SET = None

# =============================================================================
# Process
# =============================================================================
image_folder_path = Path(IMAGE_FOLDER)
output_folder_path = Path(OUTPUT_FOLDER)
metadata = None

# Reading JSON metadata
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)

if SAMPLING:
    images = os.listdir(image_folder_path)
    IMAGES_SET = frozenset(images)

# Dropping irrelevant captions
for item in metadata:
    item['captions'] = [caption for caption in item['captions'] if caption['confidence'] >= CONFIDENCE_THRESHOLD]

# Dropping irrelevant images
if IMAGES_SET is not None:
    metadata = [item for item in metadata if item['file'] in IMAGES_SET]

# Reading tweets if needed
retweets = Counter()
if TWEETS_PATH is not None:
    with open(TWEETS_PATH, 'r') as f:
        reader = csv.DictReader(f)

        for tweet in reader:
            count = int(tweet['retweet_count'])

            if not count:
                continue

            for img in tweet['medias_files'].split('|'):
                retweets[img] = count

# Creating output folder
os.makedirs(output_folder_path, exist_ok=True)

# Method n°1 - tfidf token selection
# NOTE: method n°1 is inconclusive
dfs = Counter()
for item in metadata:
    captions = item['captions']

    for caption in captions:
        text = caption['caption']
        tokens = tokenize(text)

        for token in tokens:
            dfs[token] += 1

# for item in metadata:
#     captions = item['captions']

#     for caption in captions:
#         text = caption['caption']
#         tokens = tokenize(text)
#         best_token = max(tokens, key=lambda token: math.log(1 / dfs[token]))

#         print('%s -> best token is: %s' % (colored(text, 'cyan'), colored(best_token, 'red')))

# Method n°2 - prefix clustering
trie = Trie()
for item in metadata:
    captions = item['captions']

    for caption in captions:
        text = caption['caption']
        signature = extract_signature(text, dfs, reverse=DESC_TFIDF)

        if len(signature) <= SIGNATURE_THRESHOLD:
            continue

        trie.add(signature)

prefixes = {}
prefixes_freq = Counter()
clusters = defaultdict(set)
for item in metadata:
    captions = item['captions']

    for caption in captions:
        text = caption['caption']
        signature = extract_signature(text, dfs, reverse=DESC_TFIDF)

        if len(signature) <= SIGNATURE_THRESHOLD:
            prefix = signature
        else:
            prefix = trie.shortest_prefix(signature)

        prefix = ' '.join(prefix)

        prefixes[text] = prefix
        prefixes_freq[prefix] += max(1, retweets[item['file']])
        clusters[prefix].add(text)

        # print('%s => %s' % (colored(text, 'cyan'), colored(' '.join(prefix), 'green')))

# Performing weighted sample
AUTHORIZED_PREFIXES = None

if SAMPLING:
    most_common = prefixes_freq.most_common(1)[0]

    print(
        'Most frequent prefix is %s, appearing %s times%s.' % (
            colored(most_common[0], 'cyan'),
            colored(str(most_common[1]), 'red'),
            ' (retweets taken into account)' if TWEETS_PATH is not None else ''
        )
    )

    prefix_freq_threshold = np.percentile(list(prefixes_freq.values()), PERCENTILE_THRESHOLD)
    weighted_items = []

    for prefix, freq in prefixes_freq.items():
        weighted_items.append({
            'prefix': prefix,
            'weight': prefix_freq_threshold if freq < prefix_freq_threshold else freq
        })

    AUTHORIZED_PREFIXES = frozenset(weighted_sample(weighted_items, SAMPLE_SIZE))

if LOG_CLUSTERS:
    for prefix, items in clusters.items():
        if len(items) < 2:
            continue

        print('Gathered %s for %s:' % (colored(len(items), 'red'), colored(prefix, 'cyan')))

        for text in items:
            print('   %s' % text)

        print()

    sys.exit(0)

# Processing images
bar = ProgressBar(max_value=len(metadata))
i = 0
for item in bar(iter(metadata)):

    if OFFSET is not None and i < OFFSET:
        i += 1
        continue

    captions = item['captions']
    file = item['file']
    folder = item['folder']

    # print('Processing %s (%i relevant captions)' % (file, len(captions)))
    if SAMPLING:
        p = image_folder_path / file
    else:
        p = image_folder_path / folder / file

    img = Image.open(p)
    img.thumbnail((800, 800))
    prefixes_count = Counter()

    for caption in captions:
        text = caption['caption']
        prefix = prefixes[text]
        bb = caption['bounding_box']

        if AUTHORIZED_PREFIXES is not None and prefix not in AUTHORIZED_PREFIXES:
            continue

        bb = (
            int(bb[0]),
            int(bb[1]),
            int(bb[2]),
            int(bb[3])
        )

        bb = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])

        os.makedirs(output_folder_path / prefix, exist_ok=True)
        cropped = img.crop(bb)

        try:
            cropped.save(output_folder_path / prefix / ('%i-%s' % (prefixes_count[prefix], file)))
        except:
            print('Error with %s/%s' % (folder, file))

        prefixes_count[prefix] += 1

    i += 1

    if LIMIT is not None and i == LIMIT:
        break
