# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:15:23 2021

@author: mhrfk
"""
import pandas as pd
import re
import os
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
import nltk
porter = PorterStemmer()


TRAIN_PATH = "./TRAIN/"
VAL_PATH = "./VAL/"


def createDataset2(train: bool):

    files = os.listdir(TRAIN_PATH if train else VAL_PATH)
    # print(files)
    dataset = []
    df = pd.DataFrame([[[""], "N"]], columns=["text", "c"])
    for file in files:
        match = re.search(r'(\w+)_(\w)\.txt', file)

        if match == None:
            continue
        docId = match.group(1)
        c = match.group(2)
        # print(docId,c)

        with open(TRAIN_PATH + file if train else VAL_PATH + file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            # text = getTokens(text)
            df2 = pd.DataFrame([[text, c]], columns=["text", "c"])

            df = df.append(df2)
            # print(df2)

    df = df.iloc[1:, :]
    return df


def clean(text):

    # text = p.clean(text)
    text = re.sub(r'\W+', ' ', text)  # remove non-alphanumeric characters
    # replace numbers with the word 'number'
    text = re.sub(r"\d+", "number", text)
    # don't consider sentenced with less than 3 words (i.e. assumed noise)
    if len(text.strip().split()) < 3:
        return None
    text = text.lower()  # lower case everything

    return text.strip()  # remove redundant spaces


def caseFolding(tokens: list):
    newTokens = [token.lower() for token in tokens]
    return newTokens


def punctuationRemoval(tokens: list):
    # map punctuation to space
    translator = str.maketrans(
        string.punctuation, ' ' * len(string.punctuation))
    newTokens = [token.translate(translator)
                 for token in tokens]  # Punctuation Removal
    # Retokenize the tokens that include any whitespaces after punctuation removal.
    str_tokens = " ".join(token for token in newTokens)
    newTokens = str_tokens.split()
    return newTokens


def stopwordRemoval(tokens: list):
    stopwords = get_stopwords()
    newTokens = [token for token in tokens if token not in stopwords]
    return newTokens


def get_stopwords():
    return stopwords.words('english')


def download_stopwords():
    nltk.download()


def normalization(tokens: list):
    tokens = caseFolding(tokens)
    tokens = punctuationRemoval(tokens)
    tokens = stopwordRemoval(tokens)
    tokens = numericRemoval(tokens)
    tokens = shortTokenRemoval(tokens)
    return tokens


def shortTokenRemoval(tokens: list):
    newTokens = [token for token in tokens if len(token) > 2]
    return newTokens


def numericRemoval(tokens: list):  # Preserves keywords!
    newTokens = [x for x in tokens if not any(c.isdigit() for c in x)]
    return newTokens


def getTokens(text: str):
    textTokens = text.split()

    textTokens = normalization(textTokens)

    return textTokens
