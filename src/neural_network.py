

# IMPORTING DEPENDENCIES
# system tools
import argparse
import os
# To save models and vectorizers
from joblib import dump, load
# Machine learning stuff
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import warnings # For ignoring warnings.
warnings.filterwarnings("ignore") # Ignore warnings.

def input_parse(): # Function to parse command line arguments.
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--nodes_layers", help="Number of nodes per layer. Default is one layer of 5 nodes. For one layer, do not include comma at end! NB: NO SPACES ALLOWED", type=str, default="5")
    parser.add_argument("--max_iter", help="Number of iterations. Default is 1000.", type=int, default=1000)
    # parse the arguments from the command line
    args = parser.parse_args()
    # get the name
    return args


def NN(nodes_layers_arg, max_iter_arg):
    # Load data from joblib files in "in" folder
    X_train_feats = load(os.path.join(os.getcwd(), "in", "X_train_feats.joblib"))
    X_test_feats = load(os.path.join(os.getcwd(), "in", "X_test_feats.joblib"))
    y_train = load(os.path.join(os.getcwd(), "in", "y_train.joblib"))
    y_test = load(os.path.join(os.getcwd(), "in", "y_test.joblib"))

    # initiate classifier
    classifier = MLPClassifier(activation = "logistic", # logistic because we want 2 class output
                           hidden_layer_sizes = tuple([int(i) for i in nodes_layers_arg.split(",")]), # (nodes_layer1, nodes_layer2)
                           max_iter=max_iter_arg, # updates state (W & B) 1000 times 
                           random_state = 420) # reproduceability

    # fit classifier
    classifier.fit(X_train_feats, y_train)

    # Predict
    y_pred = classifier.predict(X_test_feats)
    
    # Save classifier to out folder
    dump(classifier, os.path.join(os.getcwd(), "models", "NN_classifier.joblib"))

    # get classification report
    report = metrics.classification_report(y_true = y_test,
                                y_pred = y_pred,
                                labels = ["FAKE", "REAL"])

    # save and print report
    with open(os.path.join(os.getcwd(), "out", "NN_report.txt"), "w") as f:
        f.write(report)

    print("Neural network classification report:")
    print(report)


# This step seems highly redundant, but it's good practice.
def main():
    args = input_parse()
    NN(args.nodes_layers, args.max_iter)

if __name__ == "__main__":
    main()