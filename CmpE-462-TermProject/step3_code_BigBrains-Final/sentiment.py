from typing import List
from comment import Comment
import utils


class SentimentClass():
    def __init__(self, name: str):
        self.name = name
        # This stores all documents related to this class
        self.document_set = set()
        # This stores all words related to this class
        self.bag_of_words: dict[str, int] = {}
        # This stores document list of each word.
        self.word_document_list: dict[int, List] = {}
        # This stores document set of each word.
        self.word_document_set: dict[int, set] = {}

        # This stores all words in headers of comments related to this class
        self.header_bag_of_words: dict[str, int] = {}
        # This stores document list of each word in headers.
        self.header_word_document_list: dict[int, List] = {}
        # This stores document set of each word in headers of comments.
        self.header_word_document_set: dict[int, set] = {}

        # This stores all words in bodies of comments related to this class
        self.body_bag_of_words: dict[str, int] = {}
        # This stores document list of each word in bodies.
        self.body_word_document_list: dict[int, List] = {}
        # This stores document set of each word in bodies of comments.
        self.body_word_document_set: dict[int, set] = {}

        # These are for evaluation
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

    def __repr__(self) -> str:
        return f"{self.name} - {len(self.document_set)}"

    def add_comment(self, comment: Comment) -> None:
        comment_id = comment.number
        self.document_set.add(comment_id)

        for token in comment.content:
            # Add token into bag
            if token in self.bag_of_words:
                self.bag_of_words[token] += 1
            else:
                self.bag_of_words[token] = 1

            # Increase document frequency
            if token in self.word_document_list:
                self.word_document_list[token].append(comment_id)
            else:
                self.word_document_list[token] = [comment_id]

            # Increase document frequency
            if token in self.word_document_set:
                self.word_document_set[token].add(comment_id)
            else:
                self.word_document_set[token] = {comment_id}

        # Process tokens in headers
        for token in comment.header:
            # Add token into bag
            if token in self.header_bag_of_words:
                self.header_bag_of_words[token] += 1
            else:
                self.header_bag_of_words[token] = 1

            # Increase document frequency
            if token in self.header_word_document_list:
                self.header_word_document_list[token].append(comment_id)
            else:
                self.header_word_document_list[token] = [comment_id]

            # Increase document frequency
            if token in self.header_word_document_set:
                self.header_word_document_set[token].add(comment_id)
            else:
                self.header_word_document_set[token] = {comment_id}

        # Process tokens in bodies
        for token in comment.body:
            # Add token into bag
            if token in self.body_bag_of_words:
                self.body_bag_of_words[token] += 1
            else:
                self.body_bag_of_words[token] = 1

            # Increase document frequency
            if token in self.body_word_document_list:
                self.body_word_document_list[token].append(comment_id)
            else:
                self.body_word_document_list[token] = [comment_id]

            # Increase document frequency
            if token in self.body_word_document_set:
                self.body_word_document_set[token].add(comment_id)
            else:
                self.body_word_document_set[token] = {comment_id}

    def print_evaluation_stats(self, out_file):
        if out_file:
            print("Sentiment: '{}', TP: {}, TN: {}, FP: {}, FN: {}, Accuracy: {} Precision: {}, Recall: {}, F-Measure: {}".format(
                self.name, self.TP, self.TN, self.FP, self.FN, self.get_accuracy(
                ), self.get_precision(), self.get_recall(), self.get_f_measure()
            ), file=out_file)
        else:
            print("Sentiment: '{}', TP: {}, TN: {}, FP: {}, FN: {}, Accuracy: {} Precision: {}, Recall: {}, F-Measure: {}".format(
                self.name, self.TP, self.TN, self.FP, self.FN, self.get_accuracy(
                ), self.get_precision(), self.get_recall(), self.get_f_measure()
            ))

    # Evaluation functions

    def get_accuracy(self):
        return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)

    def get_precision(self):
        return self.TP / (self.TP + self.FP)

    def get_recall(self):
        return self.TP / (self.TP + self.FN)

    def get_f_measure(self):
        return (2 * self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall())
