from src.data.preprocess import clean_text, encode_label

def test_clean_text_removes_urls():
    assert "http://example.com" not in clean_text("check http://example.com out")

def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == "hello world"

def test_clean_text_strips_whitespace():
    assert clean_text("  hello   world  ") == "hello world"

def test_encode_label_positive():
    assert encode_label("positive") == 2

def test_encode_label_negative():
    assert encode_label("negative") == 0

def test_encode_label_neutral():
    assert encode_label("neutral") == 1

def test_encode_label_unknown():
    assert encode_label("gibberish") == -1
