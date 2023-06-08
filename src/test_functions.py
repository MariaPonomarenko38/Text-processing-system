from functions import *
import pytest
from collections import Counter
import itertools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from photo_scanning import *
from predict import *

@pytest.fixture
def lda():
    data = [
        "This is document 1",
        "Another document here",
        "Document number three",
        "Final document",
        "This is document 1",
        "Another document here",
        "Document number three",
        "Final document",
        "This is document 1",
        "Another document here",
        "Document number three",
        "Final document"
    ]
    return LDA(data)

def test_generate_frequencies(lda):
    freqs = lda.generate_frequencies(lda.data)
    assert isinstance(freqs, Counter)
    assert len(freqs) > 0

def test_get_vocab(lda):
    freqs = lda.generate_frequencies(lda.data)
    vocab, vocab_idx_str = lda.get_vocab(freqs)
    assert isinstance(vocab, dict)
    assert isinstance(vocab_idx_str, dict)
    assert len(vocab) > 0
    assert len(vocab_idx_str) > 0

def test_tokenize_dataset(lda):
    freqs = lda.generate_frequencies(lda.data)
    vocab, _ = lda.get_vocab(freqs)
    docs, corpus = lda.tokenize_dataset(lda.data, vocab)
    assert isinstance(docs, list)
    assert isinstance(corpus, list)
    assert len(docs) == len(corpus)
    assert all(isinstance(doc, list) for doc in docs)
    assert all(isinstance(doc, np.ndarray) for doc in corpus)

@pytest.fixture
def image():
    return np.ones((100, 100, 3), dtype=np.uint8) * 255

@pytest.fixture
def pts():
    return np.array([(10, 10), (90, 10), (90, 90), (10, 90)], dtype=np.float32)

def test_order_points(pts):
    expected_rect = np.array([(10, 10), (90, 10), (90, 90), (10, 90)], dtype=np.float32)
    rect = order_points(pts)
    np.testing.assert_array_equal(rect, expected_rect)

def test_four_point_transform(image, pts):
    transformed_image = four_point_transform(image, pts)
    assert transformed_image.shape == (80, 80, 3)

def test_name_entity_recognition():
    sentence = "Napoleon went to France at 13:00"

    ner_result = find_named_entities(sentence)

    assert ner_result['France'] == 'B-geo'