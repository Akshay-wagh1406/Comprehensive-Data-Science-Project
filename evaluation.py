from sklearn.metrics import accuracy_score

def evaluate_model(y_test, predictions):

    accuracy = accuracy_score(y_test, predictions)

    return accuracy