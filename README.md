# Binary Classifier

This project focuses on performing binary classification on a dataset with 1000 points, 12 features, and labels of 1 or -1 for each point. The goal is to compare the performance of simple models to complex models using ROC/AUC diagrams and identify the best model.

## Requirements

- Python 3.11.4
- Libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
```
git clone https://github.com/alirezadamash/binary-classifier.git
```
2. Navigate to the project directory:
```
cd binary-classifier
```

3. Install the required libraries:
```
pip install -r requirements.txt
```
## Usage

1. Place your dataset in a CSV file named `data.csv` in the project directory.

2. Run the main script:
```
python models.py
```
3. The ROC curve will be displayed, comparing the performance of several models. The model with the highest AUC score is considered the best.

## Dataset

The dataset used in this project contains 1000 points with 12 features. The labels are binary, with values of 1 or -1 for each point.

## Models

The following models are evaluated in this project:

- Logistic Regression
- Support Vecror Machine
- Decision Tree
- Random Forest
- Gradient Boosting

## Results

The performance of each model is compared using ROC/AUC diagrams. The model with the highest AUC score is considered the best.
