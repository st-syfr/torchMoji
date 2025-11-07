import json
from pathlib import Path

import numpy as np
import pytest

import test_helper

from torchmoji.class_avg_finetuning import relabel
from torchmoji.sentence_tokenizer import SentenceTokenizer

from torchmoji.finetuning import (
    calculate_batchsize_maxlen,
    freeze_layers,
    change_trainable,
    finetune,
    load_benchmark,
)
from torchmoji.model_def import (
    torchmoji_transfer,
    torchmoji_feature_encoding,
    torchmoji_emojis,
)
from torchmoji.global_variables import (
    PRETRAINED_PATH,
    NB_TOKENS,
    VOCAB_PATH,
    ROOT_PATH,
)

VOCAB_FILE = Path(VOCAB_PATH)
WEIGHTS_FILE = Path(PRETRAINED_PATH)
DATASET_FILE = Path(ROOT_PATH) / 'data' / 'SS-Youtube' / 'raw.pickle'

requires_weights = pytest.mark.skipif(
    not WEIGHTS_FILE.exists(),
    reason="Pretrained torchMoji weights are not available.",
)
requires_dataset = pytest.mark.skipif(
    not DATASET_FILE.exists(),
    reason="SS-Youtube dataset is not available.",
)


def _load_vocab():
    with VOCAB_FILE.open('r', encoding='utf-8') as f:
        return json.load(f)


def test_calculate_batchsize_maxlen():
    """Batch size and max length are calculated properly."""
    texts = ['a b c d', 'e f g h i']
    batch_size, maxlen = calculate_batchsize_maxlen(texts)

    assert batch_size == 250
    assert maxlen == 10, maxlen


def test_freeze_layers():
    """Correct layers are frozen."""
    model = torchmoji_transfer(5)
    keyword = 'output_layer'

    model = freeze_layers(model, unfrozen_keyword=keyword)

    for name, module in model.named_children():
        trainable = keyword.lower() in name.lower()
        assert all(p.requires_grad == trainable for p in module.parameters())


def test_change_trainable():
    """change_trainable() changes trainability of layers."""
    model = torchmoji_transfer(5)
    change_trainable(model.embed, False)
    assert not any(p.requires_grad for p in model.embed.parameters())
    change_trainable(model.embed, True)
    assert all(p.requires_grad for p in model.embed.parameters())


def test_torchmoji_transfer_extend_embedding():
    """Defining torchmoji with extension adjusts embedding size."""
    extend_with = 50
    model = torchmoji_transfer(5, weight_path=None, extend_embedding=extend_with)
    embedding_layer = model.embed
    assert embedding_layer.weight.size()[0] == NB_TOKENS + extend_with


@requires_weights
def test_torchmoji_return_attention():
    seq_tensor = np.array([[1]])
    model = torchmoji_emojis(weight_path=PRETRAINED_PATH)
    assert len(model(seq_tensor)) == 1
    model = torchmoji_emojis(weight_path=PRETRAINED_PATH, return_attention=True)
    assert len(model(seq_tensor)) == 2


def test_relabel():
    """relabel() works with multi-class labels."""
    nb_classes = 3
    inputs = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
    ])
    expected_0 = np.array([True, False, True])
    expected_1 = np.array([False, True, False])
    expected_2 = np.array([False, False, True])

    assert np.array_equal(relabel(inputs, 0, nb_classes), expected_0)
    assert np.array_equal(relabel(inputs, 1, nb_classes), expected_1)
    assert np.array_equal(relabel(inputs, 2, nb_classes), expected_2)


def test_relabel_binary():
    """relabel() works with binary classification (no changes to labels)."""
    nb_classes = 2
    inputs = np.array([True, False, False])

    assert np.array_equal(relabel(inputs, 0, nb_classes), inputs)


@pytest.mark.slow
@requires_weights
@requires_dataset
def test_finetune_full():
    """finetuning using 'full'."""
    nb_classes = 2
    min_acc = 0.68

    vocab = _load_vocab()
    data = load_benchmark(str(DATASET_FILE), vocab, extend_with=10000)
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH, extend_embedding=data['added'])
    model, acc = finetune(
        model,
        data['texts'],
        data['labels'],
        nb_classes,
        data['batch_size'],
        method='full',
        nb_epochs=1,
    )

    assert acc >= min_acc


@pytest.mark.slow
@requires_weights
@requires_dataset
def test_finetune_last():
    """finetuning using 'last'."""
    nb_classes = 2
    min_acc = 0.68

    vocab = _load_vocab()
    data = load_benchmark(str(DATASET_FILE), vocab)
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    model, acc = finetune(
        model,
        data['texts'],
        data['labels'],
        nb_classes,
        data['batch_size'],
        method='last',
        nb_epochs=1,
    )

    assert acc >= min_acc


@requires_weights
def test_score_emoji():
    """Emoji predictions make sense."""
    test_sentences = [
        "I love mom's cooking",
        "I love how you never reply back..",
        "I love cruising with my homies",
        "I love messing with yo mind!!",
        "I love you and now you're just gone..",
        "This is shit",
        "This is the shit",
    ]

    expected = [
        np.array([36, 4, 8, 16, 47]),
        np.array([1, 19, 55, 25, 46]),
        np.array([31, 6, 30, 15, 13]),
        np.array([54, 44, 9, 50, 49]),
        np.array([46, 5, 27, 35, 34]),
        np.array([55, 32, 27, 1, 37]),
        np.array([48, 11, 6, 31, 9]),
    ]

    vocab = _load_vocab()
    st = SentenceTokenizer(vocab, 30)
    tokens, _, _ = st.tokenize_sentences(test_sentences)

    model = torchmoji_emojis(weight_path=PRETRAINED_PATH)
    prob = model(tokens)

    def top_elements(array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    for i, t_prob in enumerate(list(prob)):
        assert np.array_equal(top_elements(t_prob, 5), expected[i])


@requires_weights
def test_encode_texts():
    """Text encoding is stable."""
    test_sentences = [
        "I love mom's cooking",
        "I love how you never reply back..",
        "I love cruising with my homies",
        "I love messing with yo mind!!",
        "I love you and now you're just gone..",
        "This is shit",
        "This is the shit",
    ]

    maxlen = 30

    vocab = _load_vocab()
    st = SentenceTokenizer(vocab, maxlen)

    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    tokenized, _, _ = st.tokenize_sentences(test_sentences)
    encoding = model(tokenized)

    avg_across_sentences = np.around(np.mean(encoding, axis=0)[:5], 3)
    assert np.allclose(avg_across_sentences, np.array([-0.023, 0.021, -0.037, -0.001, -0.005]))
