from typing import Dict, List, Tuple
from comment import Comment
from sentiment import SentimentClass


def calculate_result_stats(
    comment_list: List[Comment],
    sentiments: Dict[str, SentimentClass]
) -> None:
    for comment in comment_list:
        sentiment_class = sentiments[comment.sentiment]
        calculated_sentiment_class = sentiments[comment.calculated_sentiment]
        other_classes = [_class for _class in sentiments.values() if _class.name != sentiment_class.name]
        if comment.sentiment == comment.calculated_sentiment:
            sentiment_class.TP += 1
            # Increase TN of other classes
            for _class in other_classes:
                _class.TN += 1
        else:
            sentiment_class.FN += 1
            calculated_sentiment_class.FP += 1


def calculate_macro_avg(class_list: Tuple[SentimentClass],):
    macro_avg_accuracy = 0
    macro_avg_precision = 0
    macro_avg_recall = 0
    macro_avg_f_measure = 0

    for _class in class_list:
        macro_avg_accuracy += _class.get_accuracy() / len(class_list)
        macro_avg_precision += _class.get_precision() / len(class_list)
        macro_avg_recall += _class.get_recall() / len(class_list)
        macro_avg_f_measure += _class.get_f_measure() / len(class_list)

    return (macro_avg_accuracy, macro_avg_f_measure, macro_avg_precision, macro_avg_recall)
