from comment import CommentReader
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import utils
from metric import get_metrics
from preprocess import clean_words_dataset
from typing import List


class W2VMeanVectorizer(object):
    def __init__(self, word2vec: dict, dimension=50):
        self.word2vec = word2vec
        self.dimension = dimension

    def transform(self, X: List[str]):
        X_clean = clean_words_dataset(X)
        return np.array([np.mean([self.word2vec[w] for w in doc.split(' ') if w in self.word2vec] or [np.zeros(self.dimension)], axis=0) for doc in X_clean])


# Encoding for sentiment values
Encoder = LabelEncoder()

TRAIN_PATH = utils.get_abs_path('./TRAIN')
VAL_PATH = utils.get_abs_path('./VAL')

train_reader = CommentReader(path=TRAIN_PATH)
documents_train, sentiments_train = train_reader.read_documents()

test_reader = CommentReader(path=VAL_PATH)
documents_test, sentiments_test = test_reader.read_documents()


with open("./glove/glove.6B.50d.txt", "r") as lines:
    w2v = {line.split()[0]: np.array([float(i) for i in line.split()[1:]])
           for line in lines}

    vectorizer = W2VMeanVectorizer(word2vec=w2v)

    X_train = vectorizer.transform(documents_train)
    X_test = vectorizer.transform(documents_test)

    Y_train = np.array(Encoder.fit_transform(sentiments_train))
    Y_test = np.array(Encoder.fit_transform(sentiments_test))

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3)
    SVM.fit(X_train, Y_train)

    Y_train_pred = SVM.predict(X_train)
    Y_test_pred = SVM.predict(X_test)

    print("Train: accuracy, precision, recall",
          get_metrics(Y_train, Y_train_pred))

    print("Test: accuracy, precision, recall",
          get_metrics(Y_test, Y_test_pred))
