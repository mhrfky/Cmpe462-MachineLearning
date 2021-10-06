from comment import CommentReader
import numpy as np
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import utils
from metric import get_metrics
from preprocess import clean_sentences_dataset
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encoding for sentiment values
Encoder = LabelEncoder()

TRAIN_PATH = utils.get_abs_path('./TRAIN')
VAL_PATH = utils.get_abs_path('./VAL')

train_reader = CommentReader(path=TRAIN_PATH)
documents_train, sentiments_train = train_reader.read_documents()

test_reader = CommentReader(path=VAL_PATH)
documents_test, sentiments_test = test_reader.read_documents()

Y_train = np.array(Encoder.fit_transform(sentiments_train))
Y_test = np.array(Encoder.fit_transform(sentiments_test))

X_train = model.encode(clean_sentences_dataset(documents_train))
X_test = model.encode(clean_sentences_dataset(documents_test))

SVM = svm.SVC()
SVM.fit(X_train, Y_train)

Y_train_pred = SVM.predict(X_train)
Y_test_pred = SVM.predict(X_test)

get_metrics(Y_train, Y_train_pred)
get_metrics(Y_test, Y_test_pred)
