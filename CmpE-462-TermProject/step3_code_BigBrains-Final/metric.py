from sklearn.metrics import accuracy_score, precision_score, recall_score


def get_metrics(Y_actual, Y_pred):
    accuracy = accuracy_score(Y_actual, Y_pred)
    precision = precision_score(Y_actual, Y_pred, average='macro')
    recall = recall_score(Y_actual, Y_pred, average='macro')

    print("Macro avg accuracy:".ljust(24), accuracy)
    print("Macro avg precision:".ljust(24), precision)
    print("Macro avg recall:".ljust(24), recall)

    return accuracy, precision, recall
