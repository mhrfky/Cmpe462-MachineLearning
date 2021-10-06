from comment import CommentReader
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
import utils
from metric import get_metrics
from preprocess import clean_sentences_dataset
import tensorflow_hub as hub

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

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

X_train = model(clean_sentences_dataset(documents_train))
X_test = model(clean_sentences_dataset(documents_test))

# SVM
SVM = svm.SVC()
SVM.fit(X_train, Y_train)

Y_train_pred = SVM.predict(X_train)
Y_test_pred = SVM.predict(X_test)

print("SVM Train: accuracy, precision, recall",
      get_metrics(Y_train, Y_train_pred))
print("SVM Test: accuracy, precision, recall",
      get_metrics(Y_test, Y_test_pred))

# Logistic Regression

LR = LogisticRegression(random_state=0).fit(X_train, Y_train)

Y_train_pred = LR.predict(X_train)
Y_test_pred = LR.predict(X_test)

print("LR Train: accuracy, precision, recall",
      get_metrics(Y_train, Y_train_pred))
print("LR Test: accuracy, precision, recall", get_metrics(Y_test, Y_test_pred))

# Gradient Boost

GB = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)

Y_train_pred = GB.predict(X_train)
Y_test_pred = GB.predict(X_test)

print("GB Train: accuracy, precision, recall",
      get_metrics(Y_train, Y_train_pred))
print("GB Test: accuracy, precision, recall",
      get_metrics(Y_test, Y_test_pred))
