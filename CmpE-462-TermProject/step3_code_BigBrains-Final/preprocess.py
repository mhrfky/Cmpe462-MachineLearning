import string
from typing import List
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

porter = PorterStemmer()
stop_words = set(stopwords.words('english'))


def remove_punctuation(text: str) -> str:
    text = text.replace("'s", "")
    for punc in string.punctuation:
        text = text.replace(punc, "")
    return text


def tokenize(text: str, preprocess=True) -> List[str]:
    if preprocess:
        text = remove_punctuation(text).lower()
        tokens = [word for word in text.split() if word not in stop_words]
        tokens = [porter.stem(word) for word in tokens]
    else:
        tokens = text.split()

    return tokens


def clean_words(text: str) -> str:
    text = text.replace('\n', '.')
    text = text.lower()
    for punc in string.punctuation:
        text = text.replace(punc, " ")
    return text


def clean_sentences(text: str) -> str:
    text = text.replace('\n', '.')
    text = text.lower()
    return text


def tokenized_words_dataset(X: List[str]) -> List[str]:
    return [' '.join(tokenize(doc)) for doc in X]

def clean_words_dataset(X: List[str]) -> List[str]:
    return [clean_words(doc) for doc in X]


def clean_sentences_dataset(X: List[str]) -> List[str]:
    return [clean_sentences(doc) for doc in X]
