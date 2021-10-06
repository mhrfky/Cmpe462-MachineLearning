import math
from typing import List, Tuple
import sentiment_types
from comment import Comment, CommentReader
from sentiment import SentimentClass
from evaluation import calculate_result_stats, calculate_macro_avg
import utils

TRAINING_DATA_PICKLE_PATH = utils.get_abs_path('./step2_model_BigBrains.pkl')


class TrainingData:
    def __init__(self, sentiments, vocabulary, vocabulary_header, vocabulary_body) -> None:
        self.sentiments = sentiments
        self.vocabulary = vocabulary
        self.vocabulary_header = vocabulary_header
        self.vocabulary_body = vocabulary_body

    def get_data(self):
        return self.sentiments, self.vocabulary, self.vocabulary_header, self.vocabulary_body


def classify_comment(
    comment: Comment,
    class_list: Tuple[SentimentClass],
    vocabulary: set,
    alpha=1,  #  Smoothing coefficient
) -> None:
    # Total number of comments from all classes
    all_docs_count = sum([len(c.document_set) for c in class_list])
    # Find the class that has the maximum class score
    max_class_score = -math.inf
    chosen_class: SentimentClass = None
    for _class in class_list:
        # Document Count: number of documents in this class
        doc_count = len(_class.document_set)

        # Probability of this class to be chosen
        class_prob = doc_count / all_docs_count
        class_prob_log = math.log(class_prob, 10)

        # Word Count: number of distinct words in this class
        class_wc = len(_class.bag_of_words)

        # Word Total Count: number of total words in this class
        class_total_wc = sum(_class.bag_of_words.values())

        # Calculate probability for each token and sum up the log values
        feature_log_sum = 0
        for token in comment.content:
            if token not in vocabulary:
                continue
            if token in _class.word_document_set:
                word_freq = len(_class.word_document_list[token])
                doc_freq = len(_class.word_document_set[token])
            else:
                word_freq = 0
                doc_freq = 0

            token_prob = ((doc_freq + alpha) / (doc_count + alpha * class_wc)) * \
                ((word_freq + alpha) / (class_total_wc + alpha * class_wc))
            log_prob = math.log(token_prob, 10)
            feature_log_sum += log_prob

        class_score = class_prob_log + feature_log_sum
        if class_score > max_class_score:
            chosen_class = _class
            max_class_score = class_score

    comment.calculated_sentiment = chosen_class.name

# This function is a version of classify_comment which also separates header and body tokens


def classify_comment2(
    comment: Comment,
    class_list: Tuple[SentimentClass],
    vocabulary_header: set,
    vocabulary_body: set,
    alpha=1,  # Smoothing coefficient
    omega=0.6  # Weight of header
) -> None:
    # Total number of comments from all classes
    all_docs_count = sum([len(c.document_set) for c in class_list])
    # Find the class that has the maximum class score
    max_class_score = -math.inf
    chosen_class: SentimentClass = None
    for _class in class_list:
        # Document Count: number of documents in this class
        doc_count = len(_class.document_set)

        # Probability of this class to be chosen
        class_prob = doc_count / all_docs_count
        class_prob_log = math.log(class_prob, 10)

        # Word Count: number of distinct words in this class
        header_class_wc = len(_class.header_bag_of_words)

        # Word Total Count: number of total words in this class
        header_class_total_wc = sum(_class.header_bag_of_words.values())

        # Calculate probability for each token and sum up the log values
        feature_log_sum = 0
        for token in comment.header:
            if token not in vocabulary_header:
                continue
            if token in _class.header_word_document_set:
                doc_freq = len(_class.header_word_document_set[token])
                word_freq = len(_class.header_word_document_list[token])
            else:
                doc_freq = 0
                word_freq = 0

            token_prob = ((doc_freq + alpha) / (doc_count + alpha * header_class_wc)) * (
                (word_freq + alpha) / (header_class_total_wc + alpha * header_class_wc))
            log_prob = math.log(token_prob, 10)
            feature_log_sum += log_prob * omega

        # Word Count: number of words in this class
        body_class_wc = len(_class.body_bag_of_words)

        # Word Total Count: number of total words in this class
        body_class_total_wc = sum(_class.body_bag_of_words.values())

        # Calculate probability for each token and sum up the log values
        for token in comment.body:
            if token not in vocabulary_body:
                continue
            if token in _class.body_word_document_set:
                doc_freq = len(_class.body_word_document_set[token])
                word_freq = len(_class.body_word_document_list[token])
            else:
                doc_freq = 0
                word_freq = 0

            token_prob = ((doc_freq + alpha) / (doc_count + alpha * body_class_wc)) * \
                ((word_freq + alpha) / (body_class_total_wc + alpha * body_class_wc))
            log_prob = math.log(token_prob, 10)
            feature_log_sum += log_prob * (1 - omega)

        class_score = class_prob_log + feature_log_sum
        if class_score > max_class_score:
            chosen_class = _class
            max_class_score = class_score

    comment.calculated_sentiment = chosen_class.name


def naive_bayes_training(train_path: str):
    train_reader = CommentReader(path=train_path)
    train_comments: List[Comment] = train_reader.read()

    # Populate sentiment classes
    sentiments = {
        sentiment_types.N: SentimentClass(sentiment_types.N),
        sentiment_types.P: SentimentClass(sentiment_types.P),
        sentiment_types.Z: SentimentClass(sentiment_types.Z)
    }

    # Train model
    for comment in train_comments:
        sentiments[comment.sentiment].add_comment(comment)

    # Create vocabulary
    vocabulary = set()
    for comment in train_comments:
        for token in comment.content:
            vocabulary.add(token)

    # Create header vocabulary
    vocabulary_header = set()
    for comment in train_comments:
        for token in comment.header:
            vocabulary_header.add(token)

    # Create body vocabulary
    vocabulary_body = set()
    for comment in train_comments:
        for token in comment.body:
            vocabulary_body.add(token)

    # Pickle results
    training_data = TrainingData(
        sentiments, vocabulary, vocabulary_header, vocabulary_body)
    utils.pickle_object(training_data, TRAINING_DATA_PICKLE_PATH)
    return training_data


def naive_bayes(train_path: str, val_path: str, pickle_path: str):
    # Try getting trained model from pickle
    if utils.file_exists(pickle_path):
        training_data: TrainingData = utils.unpickle_object(
            pickle_path)
    else:
        training_data = naive_bayes_training(
            train_path)

    sentiments, vocabulary, vocabulary_header, vocabulary_body = training_data.get_data()

    val_reader = CommentReader(path=val_path)
    val_comments: List[Comment] = val_reader.read()

    sentiment_classes = tuple(sentiments.values())

    # Classify validation comments
    for comment in val_comments:
        # TODO: Decide if we want to separate header and body, then delete unused method
        # classify_comment(comment, sentiment_classes, vocabulary)
        classify_comment2(comment, sentiment_classes,
                          vocabulary_header, vocabulary_body)

    calculate_result_stats(comment_list=val_comments, sentiments=sentiments)

    macro_avg_accuracy, macro_avg_f_measure, macro_avg_precision, macro_avg_recall = calculate_macro_avg(
        sentiment_classes)

    utils.create_dir("./out")
    with open(utils.get_abs_path('./out/logs.txt'), 'w+') as out_file:
        for sentiment_class in sentiment_classes:
            sentiment_class.print_evaluation_stats(out_file=out_file)
        print("Macro Avg Accuracy:", macro_avg_accuracy, file=out_file)
        print("Macro Avg Precision:", macro_avg_precision, file=out_file)
        print("Macro Avg Recall:", macro_avg_recall, file=out_file)
        print("Macro Avg F-Measure:", macro_avg_f_measure, file=out_file)

    for comment in val_comments:
        print(comment.calculated_sentiment, end="")

    print()
