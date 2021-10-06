from comment import CommentReader
import utils
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from metric import get_metrics
from preprocess import clean_words_dataset

np.random.seed(500)
# Encoding for sentiment values
Encoder = LabelEncoder()

TRAINING_DATA_PICKLE_PATH = utils.get_abs_path('./step3_model_BigBrains.pkl')


class TrainingData:
    model: svm.SVC
    vectorizer: TfidfVectorizer

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer


def train_svm_tfidf(train_path):
    TRAIN_PATH = utils.get_abs_path('./TRAIN')
    VAL_PATH = utils.get_abs_path('./VAL')

    train_reader = CommentReader(path=TRAIN_PATH)
    documents_train, sentiments_train = train_reader.read_documents()

    test_reader = CommentReader(path=VAL_PATH)
    documents_test, sentiments_test = test_reader.read_documents()

    tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    tfidfvectorizer.fit(documents_train)

    X_train = tfidfvectorizer.transform(clean_words_dataset(documents_train))
    Y_train = np.array(Encoder.fit_transform(sentiments_train))

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3)
    SVM.fit(X_train, Y_train)

    training_data = TrainingData(SVM, tfidfvectorizer)
    utils.pickle_object(training_data, TRAINING_DATA_PICKLE_PATH)

    return training_data


def apply_svm_tfidf(train_path: str, val_path: str, pickle_path: str):
    # Try getting trained model from pickle
    if utils.file_exists(pickle_path):
        training_data: TrainingData = utils.unpickle_object(pickle_path)
    else:
        training_data: TrainingData = train_svm_tfidf(train_path)

    model = training_data.model
    vectorizer = training_data.vectorizer

    test_reader = CommentReader(path=val_path)
    documents_test, sentiments_test = test_reader.read_documents()

    X_test = vectorizer.transform(clean_words_dataset(documents_test))
    Y_test = np.array(Encoder.fit_transform(sentiments_test))
    Y_test_pred = model.predict(X_test)

    result = Encoder.inverse_transform(Y_test_pred)
    for c in result:
        print(c, end='')
    print()
