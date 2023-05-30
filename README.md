<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h1 align="center">Cultural Data Science 2023</h1> 
  <h2 align="center">Assignment 2</h2> 
  <h3 align="center">Language Analytics</h3> 


  <p align="center">
    Aleksandrs Baskakovs
  </p>
</p>


<!-- Assignment instructions -->
## Assignment instructions

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

<!-- ABOUT THE PROJECT -->
## About the project
This repository contains three Python scripts that allow classification of text data via either logistic regression or a neural network using ```scikit-learn``` library. The scripts vectorize the data using ```tf-idf```, train the chosen model, and save the results in the ```out``` folder. The scripts also save the trained models and vectorizers in the ```models``` folder.

<!-- Data -->
## Data
The sample dataset used for this project is the Fake News Dataset. The dataset contains 10556 news articles, each labeled as either "REAL" or "FAKE". The dataset is available in the ```in``` folder. However, your own dataset can be used as well, as long as it has a similar structure - a ```.csv``` file with at least two columns, one containing the text and the other containing the label, and is placed in the ```in``` folder.

<!-- USAGE -->
## Usage
To use the code you need to adopt the following steps.

**NOTE:** Please note that the instructions provided here have been tested on a Mac machine running macOS Ventura 13.1, using Visual Studio Code version 1.76.0 (Universal) and a Unix-based bash terminal. While they should also be compatible with other Unix-based systems like Linux, slight variations may exist depending on the terminal and operating system you are using. To ensure a smooth installation process and avoid potential package conflicts, it is recommended to use the provided ```setup.sh``` bash file, which includes the necessary steps to create a virtual environment for the project. However, if you encounter any issues or have questions regarding compatibility on other platforms, please don't hesitate to reach out for assistance.

1. Clone repository
2. Run ``setup.sh`` in the terminal
3. Run ```vectorizer.py``` in the terminal
4. Run ```logistic_regression.py``` or ```neural_network.py``` in the terminal
5. Deactivate virtual environment

### Clone repository

Clone repository using the following lines in the your terminal:

```bash
git clone https://github.com/sashapustota/simple-text-classification
cd simple-text-classification
```

### Run ```setup.sh```

The ``setup.sh`` script is used to automate the installation of project dependencies and configuration of the environment. By running this script, you ensure consistent setup across different environments and simplify the process of getting the project up and running.

The script performs the following steps:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the required packages
4. Deactivates the virtual environment

To run the script, run the following line in the terminal:

```bash
bash setup.sh
```

Make sure to activate the virtual environment before continuing with the next steps by running the following line in the terminal:

```bash
source ./simple-text-classification-venv/bin/activate
```

### Run ```vectorizer.py```

The ```vectorizer.py``` script perform the following steps:

1. Loads the data from the ```in``` folder
2. Vectorizes the data using ```tf-idf```
3. Splits the data into training and test sets
4. Saves the vectorizer in the ```models``` folder
5. Saves the vectorized data in the ```in``` folder

To run the script, run the following lines in the terminal:

```bash
python3 src/vectorizer.py
```

The script has the following optional arguments:

```
vectorizer.py [-h] [--data DATA] [--text TEXT] [--label LABEL] [--test_size TEST_SIZE] [--ngram_range NGRAM_RANGE] [--lowercase LOWERCASE] [--max_df MAX_DF] [--min_df MIN_DF] [--max_features MAX_FEATURES]

options:
  -h, --help            show this help message and exit
  --data DATA           Name of data file, should be a .csv (default: fake_or_real_news.csv)
  --text TEXT           Name of text (X) column in data file (default: text)
  --label LABEL         Name of label (y) column in data file (default: label)
  --test_size TEST_SIZE
                        Size of test split, int between 0-1. (default: 0.2)
  --ngram_range NGRAM_RANGE
                        Ngram range for vectorizer, two digits seperated by a comma. NB: NO SPACES ALLOWED (default: 1,2)
  --lowercase LOWERCASE
                        If the data should be transformed to lowercase (default: True)
  --max_df MAX_DF       Specify max_df parameter (default: 0.95)
  --min_df MIN_DF       Specify min_df parameter (default: 0.05)
  --max_features MAX_FEATURES
                        Specify max number of features for vectorizer (default: 500)
```

An example below demonstrates how to run the script with custom arguments:

```bash
python3 src/vectorizer.py --data my_data.csv --text comments --label class --test_size 0.4 --ngram_range 1,3 --lowercase True --max_df 0.9 --min_df 0.1 --max_features 1000
```

### Run ```logistic_regression.py``` or ```neural_network.py```

The ```logistic_regression.py``` and ```neural_network.py``` scripts perform the following steps:

1. Loads the vectorized train and test data from the ```in``` folder
2. Trains the chosen model
3. Saves the trained model in the ```models``` folder
4. Saves the results in the ```out``` folder
5. Prints the results in the terminal

For ```logistic regression```, run the following line in the terminal:

```bash
python3 src/logistic_regression.py
```

For ```neural network```, run the following line in the terminal:

```bash
python3 src/neural_network.py
```

The neural network script has the following optional arguments:

```
neural_network.py [-h] [--nodes_layers NODES_LAYERS] [--max_iter MAX_ITER]

options:
  -h, --help            show this help message and exit
  --nodes_layers NODES_LAYERS
                        Number of nodes per layer. Default is one layer of 5 nodes. For one layer, do not include comma at end! NB: NO SPACES ALLOWED (default: 5)
  --max_iter MAX_ITER   Number of iterations. Default is 1000. (default: 1000)
```

An example below demonstrates how to run the script with custom arguments:

```bash
python3 src/neural_network.py --nodes_layers 10,5,3 --max_iter 500
```

### Deactivate virtual environment

When you are done running the scripts, deactivate the virtual environment by running the following line in the terminal:

```bash
deactivate
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure

This repository has the following structure:
```
│   .gitignore
│   README.md
│   requirements.txt
│   setup.sh
│       
├───in
│       fake_or_real_news.csv
│       
├───models
│       .gitkeep
│       
├───out
│       .gitkeep
│
└───src
        .gitkeep
        logistic_regression.py
        neural_network.py
        vectorizer.py
```

<!-- RESULTS -->
## Results
The following results were obtained with the default settings for both classification methods:

```
Logistic regression classification report:
              precision    recall  f1-score   support

        FAKE       0.89      0.87      0.88       634
        REAL       0.87      0.90      0.88       633

    accuracy                           0.88      1267
   macro avg       0.88      0.88      0.88      1267
weighted avg       0.88      0.88      0.88      1267
```
```
Neural network classification report:
              precision    recall  f1-score   support

        FAKE       0.88      0.88      0.88       634
        REAL       0.88      0.88      0.88       633

    accuracy                           0.88      1267
   macro avg       0.88      0.88      0.88      1267
weighted avg       0.88      0.88      0.88      1267
```
