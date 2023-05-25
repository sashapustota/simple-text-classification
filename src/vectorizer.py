# IMPORTING DEPENDENCIES
# system tools
import os
import argparse
# To save models and vectorizers
from joblib import dump
# Machine learning stuff
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings # For ignoring warnings.
warnings.filterwarnings("ignore") # Ignore warnings.

def input_parse(): # Function to parse command line arguments.
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--data", help="Name of data file, should be a .csv", type=str, default="fake_or_real_news.csv")
    parser.add_argument("--text", help="Name of text (X) column in data file", type=str, default="text")
    parser.add_argument("--label", help="Name of label (y) column in data file", type=str, default="label")
    parser.add_argument("--test_size", help="Size of test split, int between 0-1.", type=float, default=0.2)
    parser.add_argument("--ngram_range", help="Ngram range for vectorizer, two digits seperated by a comma. NB: NO SPACES ALLOWED", type=str, default="1,2")
    parser.add_argument("--lowercase", help="If the data should be transformed to lowercase", type=bool, default=True)
    parser.add_argument("--max_df", help="Specify max_df parameter", type=float, default=0.95)
    parser.add_argument("--min_df", help="Specify min_df parameter", type=float, default=0.05)
    parser.add_argument("--max_features", help="Specify max number of features for vectorizer", type=int, default=500)
    # parse the arguments from the command line
    args = parser.parse_args()
    # get the name
    return args

def load_data(data_arg, text_arg, label_arg, test_size_arg):
    filename = os.path.join(os.getcwd(), "in", data_arg) # load the data
    data = pd.read_csv(filename, index_col=0)
    
    X = data[text_arg]
    y = data[label_arg]
    
    # add your train/test split code here
    X_train, X_test, y_train, y_test = train_test_split(X,           # texts for the model
                                                        y,          # classification labels
                                                        test_size=float(test_size_arg),   # create an 80/20 split
                                                        random_state=420) # "random" state for reproducibility
    
    return X_train, X_test, y_train, y_test

def vectorize(ngram_range_arg, lowercase_arg, max_df_arg, min_df_arg, max_features_arg):
    # add your vectorizer code here    
    vectorizer = TfidfVectorizer(ngram_range = tuple([int(i) for i in ngram_range_arg.split(",")]),     # unigrams and bigrams (1 word and 2 word units)
                                lowercase = lowercase_arg,       # why use lowercase?
                                max_df = max_df_arg,           # remove very common words
                                min_df = min_df_arg,           # remove very rare words
                                max_features = max_features_arg)      # max features
    
    return vectorizer

def fit(X_train, X_test, vectorizer):
    # first we fit the vectorizer to the training data...
    X_train_feats = vectorizer.fit_transform(X_train)

    #... then transform our test data
    X_test_feats = vectorizer.transform(X_test)
    
    return X_train_feats, X_test_feats

def multidump(vectorizer, X_train_feats, X_test_feats, y_train, y_test):
    dump(vectorizer, os.path.join(os.getcwd(), "models", "vectorizer.joblib"))
    dump(X_train_feats, os.path.join(os.getcwd(), "in", "X_train_feats.joblib"))
    dump(X_test_feats, os.path.join(os.getcwd(), "in", "X_test_feats.joblib"))
    dump(y_train, os.path.join(os.getcwd(), "in", "y_train.joblib"))
    dump(y_test, os.path.join(os.getcwd(), "in", "y_test.joblib"))

# Define main function
def main():
    args = input_parse()
    X_train, X_test, y_train, y_test = load_data(args.data, args.text, args.label, args.test_size)
    vectorizer = vectorize(args.ngram_range, args.lowercase, args.max_df, args.min_df, args.max_features)
    X_train_feats, X_test_feats = fit(X_train, X_test, vectorizer)
    multidump(vectorizer, X_train_feats, X_test_feats, y_train, y_test)
    print("Vectorizer saved to models folder.")

if __name__=="__main__":
    main()