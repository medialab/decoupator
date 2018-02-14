# =============================================================================
# Decoupator helpers
# =============================================================================
#
# Miscellaneous helpers.
#
import math
from stop_words import get_stop_words
from bisect import bisect_right
from random import random

STOP_WORDS = frozenset(get_stop_words('english'))

# =============================================================================
# Helpers
# =============================================================================
def tokenize(caption, stop_words=True):
    tokens = caption.strip().lower().split(' ')

    tokens = [token for token in tokens if token == 'building' or not token.endswith('ing')]

    if not stop_words:
        return tokens

    return [token for token in tokens if token not in STOP_WORDS]

def extract_signature(caption, dfs, reverse=False):
    tokens = set(tokenize(caption))

    return sorted(tokens, key=lambda token: math.log(1 / dfs[token]), reverse=reverse)

def weighted_sample(items, n):

    total = 0
    cdf = []

    for item in items:
        weight = item['weight']
        total += weight
        cdf.append(total)

    sample = set()

    while len(sample) != n:
        r = random() * total
        sample.add(bisect_right(cdf, r))

    return [items[i]['prefix'] for i in sample]

# =============================================================================
# Trie
# =============================================================================
class TrieNode(object):
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.leaf = False

class Trie(object):
    def __init__(self):
        self.root = {}

    def add(self, tokens):
        children = self.root
        node = None

        for token in tokens:
            if token not in children:
                node = TrieNode(token)
                children[token] = node
            else:
                node = children[token]

            children = children[token].children

        node.leaf = True

    def dfs(self):
        stack = [(node, 0) for node in self.root.values()]

        while len(stack):
            node, level = stack.pop()

            yield node, level

            for child in node.children.values():
                stack.append((child, level + 1))

    def shortest_prefix(self, string):
        children = self.root
        prefix = []

        for i in range(len(string)):
            token = string[i]

            if token not in children:
                break

            node = children[token]
            prefix.append(token)

            if node.leaf:
                break

            children = node.children

        return prefix
