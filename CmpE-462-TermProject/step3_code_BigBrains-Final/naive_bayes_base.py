import os
import string
import math
import re
from nltk.stem import PorterStemmer

porter = PorterStemmer()
# flags:
puncRemoval = False

TRAIN_PATH = "./TRAIN/"
VAL_PATH = "./VAL/"
# vocabularies


# variables
alpha = 1
omega = .4

numberOfTotal = 0  # number of total mails


class NaiveBayesBaseTrainer:

    def __init__(self):
        self.voc = {"N": {}, "P": {}, "Z": {}}
        self.headerVoc = {"N": {}, "P": {}, "Z": {}}
        self.distVoc = {"N": {}, "P": {}, "Z": {}}
        self.distHeaderVoc = {"N": {}, "P": {}, "Z": {}}
        self.numberOfEach_c = {"N": 0, "P": 0, "Z": 0}

        self.training()

    def training(self):
        files = os.listdir(TRAIN_PATH)
        for file in files:
            c = re.search(r'.*_(.)\.txt', file).group(1)

            self.numberOfEach_c[c] += 1

            with open(TRAIN_PATH + file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

                a = self.preprocess(c, text)
                if a == 0:
                    print(file)

        self.numberOfTotal = self.numberOfEach_c["P"] + \
            self.numberOfEach_c["N"] + self.numberOfEach_c["Z"]

        self.length = len(list(set(list(self.voc["P"].keys(
        )) + list(self.voc["N"].keys()) + list(self.voc["Z"].keys()))))
        self.pWords = sum(
            y for x, y in self.voc["P"].items())+alpha * self.length
        self.nWords = sum(
            y for x, y in self.voc["N"].items())+alpha * self.length
        self.zWords = sum(
            y for x, y in self.voc["Z"].items())+alpha * self.length

    def preprocess(self, c: str, text: str):

        headerTokens, textTokens = getTokens(text)
        if headerTokens == 0:
            return
        distHeader = list(set(headerTokens))
        distText = list(set(textTokens))

        for token in distHeader:
            if token in self.distHeaderVoc[c]:
                self.distHeaderVoc[c][token] += 1
            else:
                self.distHeaderVoc[c][token] = 1

        for token in headerTokens:
            if token in self.headerVoc[c]:
                self.headerVoc[c][token] += 1
            else:
                self.headerVoc[c][token] = 1

        for token in distText:
            if token in self.distVoc[c]:
                self.distVoc[c][token] += 1
            else:
                self.distVoc[c][token] = 1

        for token in textTokens:
            if token in self.voc[c]:
                self.voc[c][token] += 1
            else:
                self.voc[c][token] = 1

    def testFolder(self):
        global t, f, total
        files = os.listdir(VAL_PATH)
        t = 0
        f = 0
        total = 0
        for file in files:
            c = re.search(r'.*_(.)\.txt', file).group(1)
            text = ""
            cEval = 0
            with open(VAL_PATH + file, "r", encoding="utf-8", errors="ignore") as fi:
                text = fi.read()

            cEval = self.isLegitimate(text)

            total += 1
            if cEval != 0:
                if cEval == c:
                    t += 1
                else:
                    f += 1
            else:
                print(file)
        print(t, f, total)

    def isLegitimate(self, text: str):

        pPoint = 0
        nPoint = 0
        zPoint = 0

        headerTokens, textTokens = getTokens(text)

        if headerTokens == 0:

            return 0

        # print(legWords,spamWords)
        for token in textTokens:
            if token in self.voc["P"]:
                pay = (alpha + self.voc["P"][token])
            else:
                pay = alpha

            pPoint += math.log(pay / self.pWords)

            if token in self.voc["N"]:
                pay = (alpha + self.voc["N"][token])
            else:
                pay = alpha

            nPoint += math.log(pay / self.nWords)

            if token in self.voc["Z"]:
                pay = (alpha + self.voc["Z"][token])
            else:
                pay = alpha

            zPoint += math.log(pay / self.zWords)

        pPoint *= omega
        nPoint *= omega
        zPoint *= omega

        tpPoint = 0
        tnPoint = 0
        tzPoint = 0

        for token in headerTokens:
            if token in self.headerVoc["P"]:
                pay = (alpha + self.headerVoc["P"][token])
            else:
                pay = alpha

            tpPoint += math.log(pay / self.pWords)

            if token in self.headerVoc["N"]:
                pay = (alpha + self.headerVoc["N"][token])
            else:
                pay = alpha

            tnPoint += math.log(pay / self.nWords)

            if token in self.headerVoc["Z"]:
                pay = (alpha + self.headerVoc["Z"][token])
            else:
                pay = alpha

            tzPoint += math.log(pay / self.zWords)

        pPoint += math.log(self.numberOfEach_c["P"] /
                           self.numberOfTotal) + tpPoint * (1-omega)
        nPoint += math.log(self.numberOfEach_c["N"] /
                           self.numberOfTotal) + tnPoint * (1-omega)
        zPoint += math.log(self.numberOfEach_c["Z"] /
                           self.numberOfTotal) + tzPoint * (1-omega)

        if pPoint > zPoint:
            if pPoint > nPoint:
                return "P"
            else:
                return "N"
        else:
            if zPoint > nPoint:
                return "Z"
            else:
                return "N"


def puncRemoval(text):
    text = text.replace("'s", "")
    for punc in string.punctuation:
        text = text.replace(punc, "")
    return text

    # distVoc.pop('')
    # headerVoc.pop('')
    # distHeaderVoc.pop('')
    # voc.pop('')


def getTokens(text: str):
    text1 = puncRemoval(text)
    text2 = text1.lower()
    header = ""
    text = ""
    try:
        reg = re.search(r'(.*)\n(.*)', text2)
        header = reg.group(1)
        text = reg.group(2)
    except:
        print("come on-", text)
        return 0, 0

    headerTokens = header.split()
    textTokens = text.split()
    headerTokens = [porter.stem(word) for word in headerTokens]
    textTokens = [porter.stem(word) for word in textTokens]

    return headerTokens, textTokens


model = NaiveBayesBaseTrainer()
model.testFolder()
