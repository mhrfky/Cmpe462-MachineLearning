# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:12:10 2021

@author: mhrfk
"""

from utils2 import clean, createDataset2
import tensorflow as tf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVC
# we use spacy's list of stop words to clean our data
from spacy.lang.en import stop_words as spacy_stopwords
import string
import numpy as np
import tensorflow_hub as hub
import tensorflow_text  # this needs to be imported to set up some stuff in the background
from metric import get_metrics

embed = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")


# p.set_options(p.OPT.URL, p.OPT.MENTION)  # removes mentions and URLs only
stop_words = spacy_stopwords.STOP_WORDS
punctuations = string.punctuation

df = createDataset2(True)
df = df.assign(clean_text=df.text.apply(clean)).dropna()
ttf = createDataset2(False)
ttf = ttf.assign(clean_text=ttf.text.apply(clean)).dropna()

msg_train, y_train = (df.clean_text, df.c)
msg_test, y_test = (ttf.clean_text, ttf.c)
X_test = embed(msg_test)
X_test.shape

splits = np.array_split(msg_train, 10)
l = list()
for split in splits:
    l.append(embed(split))


X_train = tf.concat(l, axis=0)
del l
X_train.shape


class_weight = compute_class_weight(
    class_weight='balanced', classes=["N", "P", "Z"], y=y_train
)


print("SVC with word-embedding")
clf = SVC(class_weight={
          "N": class_weight[0], "P": class_weight[1], "Z": class_weight[2]})
# train the model
clf.fit(X_train, y_train)
# use the model to predict the testing instances
y_pred = clf.predict(np.array(X_test))
# generate the classification report
print(get_metrics(y_test, y_pred))

print("\nRandom Forest Classifier with word-embedding")
clf = RandomForestClassifier(
    class_weight={"N": class_weight[0], "P": class_weight[1], "Z": class_weight[2]})
clf.fit(X_train, y_train)
y_pred = clf.predict(np.array(X_test))
print(get_metrics(y_test, y_pred))


print("\nLogistic Regression with word-embedding")
clf = LogisticRegression(
    class_weight={"N": class_weight[0], "P": class_weight[1], "Z": class_weight[2]})
clf.fit(X_train, y_train)
y_pred = clf.predict(np.array(X_test))
print(get_metrics(y_test, y_pred))

print("\nExtra Trees Classifier wwwwith word-embedding")
clf = ExtraTreesClassifier(
    class_weight={"N": class_weight[0], "P": class_weight[1], "Z": class_weight[2]})
clf.fit(X_train, y_train)
y_pred = clf.predict(np.array(X_test))
print(get_metrics(y_test, y_pred))
