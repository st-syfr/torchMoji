# -*- coding: utf-8 -*-
""" Global variables.
"""
import sys
import tempfile
from os.path import abspath, dirname, join

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend(['{}BLANK_{}'.format(SPECIAL_PREFIX, i) for i in range(6, 10)])


# Determine if running in a PyInstaller bundle
def _get_base_path():
    """Get the base path for resources, handling PyInstaller bundles."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running in normal Python environment
        return dirname(dirname(abspath(__file__)))


ROOT_PATH = _get_base_path()
VOCAB_PATH = join(ROOT_PATH, 'model', 'vocabulary.json')
PRETRAINED_PATH = join(ROOT_PATH, 'model', 'pytorch_model.bin')

WEIGHTS_DIR = tempfile.mkdtemp()

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64
FINETUNING_METHODS = ['last', 'full', 'new', 'chain-thaw']
FINETUNING_METRICS = ['acc', 'weighted']
