import os
from typing import List
import utils
import re
from preprocess import tokenize
import sentiment_types


class Comment:
    def __init__(self, header: List[str], body: List[str], number: int, sentiment: str):
        self.header = header
        self.body = body
        self.content = header + body
        self.sentiment = sentiment
        self.number = number
        self.calculated_sentiment: str = None

    def __repr__(self) -> str:
        return f"{str(self.number).ljust(4)} - {self.sentiment} - {self.header[:5]} - {self.body[:5]}"


class CommentReader:
    def __init__(self, path: str):
        self.path = path

    def split_header_body(self, text: str):
        reg = re.search(r"(.*)\n(.*)", text)
        header = reg.group(1)
        body = reg.group(2)

        return header, body

    def get_number_sentiment(self, training_file):
        match = re.search(r'(\d+)_(.)\.txt', training_file)
        number, sentiment = int(match.group(1)), match.group(2)

        return (number, sentiment)

    def read(self, preprocess=True) -> List[Comment]:
        sentiments = [sentiment_types.N, sentiment_types.P, sentiment_types.Z]
        comments = []
        training_files = os.listdir(self.path)
        training_files.sort(key=lambda x: self.get_number_sentiment(x)[0])
        for training_file in training_files:
            number, sentiment = self.get_number_sentiment(training_file)
            if sentiment not in sentiments:
                continue
            with open(f"{self.path}/{training_file}", "r", encoding="utf-8", errors="ignore") as file:
                text = file.read()
                try:
                    # Create comment instance
                    header, body = self.split_header_body(text)
                    header_tokens = tokenize(header, preprocess)
                    body_tokens = tokenize(body, preprocess)

                    comment = Comment(
                        header_tokens, body_tokens, number, sentiment)
                    comments.append(comment)
                except Exception as e:
                    # print("An error occured while reading comment:",
                    #       number, sentiment)
                    # print("Error message:", e)
                    pass

        return comments
