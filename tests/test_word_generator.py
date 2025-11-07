"""Tests for the WordGenerator utilities."""

import test_helper
import pytest

from torchmoji.word_generator import WordGenerator


def test_bytes_are_decoded_before_tokenization():
    wg = WordGenerator([b'Hello World!'], allow_unicode_text=True)
    words, info = next(iter(wg))
    assert words == ['hello', 'world', '!']
    assert info == {}


def test_non_text_input_raises_value_error():
    wg = WordGenerator([], allow_unicode_text=True)
    with pytest.raises(ValueError):
        wg.get_words(123)


def test_unicode_sentences_ignored_if_set():
    """Strings with Unicode characters tokenize to empty array if they're not allowed."""
    sentence = ['Dobrý den, jak se máš?']
    wg = WordGenerator(sentence, allow_unicode_text=False)
    assert wg.get_words(sentence[0]) == []


def test_check_ascii_handles_bytes_and_strings():
    wg = WordGenerator([])
    assert wg.check_ascii('ASCII')
    assert wg.check_ascii(b'ASCII')
    assert not wg.check_ascii('ščřžýá')
    assert not wg.check_ascii('❤ ☀ ☆ ☂ ☻ ♞ ☯ ☭ ☢')


def test_convert_unicode_word():
    """convert_unicode_word converts Unicode words correctly."""
    wg = WordGenerator([], allow_unicode_text=True)

    result = wg.convert_unicode_word('č')
    assert result == (True, '\u010d'), '{}'.format(result)


def test_convert_unicode_word_ignores_if_set():
    """convert_unicode_word ignores Unicode words if set."""
    wg = WordGenerator([], allow_unicode_text=False)

    result = wg.convert_unicode_word('č')
    assert result == (False, ''), '{}'.format(result)


def test_convert_unicode_chars():
    """convert_unicode_word correctly converts accented characters."""
    wg = WordGenerator([], allow_unicode_text=True)
    result = wg.convert_unicode_word('ěščřžýáíé')
    assert result == (True, '\u011b\u0161\u010d\u0159\u017e\xfd\xe1\xed\xe9'), '{}'.format(result)
