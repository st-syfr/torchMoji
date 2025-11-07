# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a single text input
"""

from __future__ import print_function, division, unicode_literals
import example_helper
import json
import csv
import argparse

import numpy as np
import emoji

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from torchmoji.emojis import EMOJI_ALIASES

# Emoji map in emoji_overview.png

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--text', type=str, required=True, help="Input text to emojize")
    argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
    args = argparser.parse_args()

    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, args.maxlen)

    # Loading model
    model = torchmoji_emojis(PRETRAINED_PATH)
    # Running predictions
    tokenized, _, _ = st.tokenize_sentences([args.text])
    # Get sentence probability
    prob = model(tokenized)[0]

    # Top emoji id
    emoji_ids = top_elements(prob, 5)

    # map to emojis
    emojis = map(lambda x: EMOJI_ALIASES[x], emoji_ids)

    print(emoji.emojize("{} {}".format(args.text,' '.join(emojis)), use_aliases=True))
