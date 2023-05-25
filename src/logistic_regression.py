# IMPORTING DEPENDENCIES
# system tools
import os
# To save models and vectorizers
from joblib import dump, load
# Machine learning stuff
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings # For ignoring warnings.
warnings.filterwarnings("ignore") # Ignore warnings.


def LR():
    # Load data from joblib files in "in" folder
    X_train_feats = load(os.path.join(os.getcwd(), "in", "X_train_feats.joblib"))
    X_test_feats = load(os.path.join(os.getcwd(), "in", "X_test_feats.joblib"))
    y_train = load(os.path.join(os.getcwd(), "in", "y_train.joblib"))
    y_test = load(os.path.join(os.getcwd(), "in", "y_test.joblib"))

    # initiate classifier
    classifier = LogisticRegression(random_state=420) # "random"

    classifier.fit(X_train_feats, y_train)

    # Predict
    y_pred = classifier.predict(X_test_feats)
    
    # Save classifier to out folder
    dump(classifier, os.path.join(os.getcwd(), "models", "LR_classifier.joblib"))

    # get classification report
    report = metrics.classification_report(y_true = y_test,
                                y_pred = y_pred,
                                labels = ["FAKE", "REAL"])

    # save and print report
    with open(os.path.join(os.getcwd(), "out", "LR_report.txt"), "w") as f:
        f.write(report)

    print("Logistic regression classification report:")
    print(report)


# This step seems highly redundant, but it's good practice.
def main():
    LR()

if __name__ == "__main__":
    main()