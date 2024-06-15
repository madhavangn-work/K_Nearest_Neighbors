# KNN Classifier Implementation

This project implements a simple K-Nearest Neighbors (KNN) classifier from scratch in Python. It includes a custom implementation of KNN (KNNClassifier) and compares its performance with KNeighborsClassifier from scikit-learn on the Iris dataset.

## Features

- **KNNClassifier:** A custom KNN classifier supporting Euclidean and Manhattan distance metrics, with options for training, testing, and evaluation.
- **Evaluation:** Computes accuracy, precision, recall, and F1-score for binary classification tasks. Also includes ROC curve plotting.
- **Comparison:** Compares performance with scikit-learn's KNeighborsClassifier using accuracy score.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: numpy, pandas, scikit-learn, matplotlib

### Installation

1. Clone the repository:

```
git clone https://github.com/madhavgn007/K_Nearest_Neighbors.git
cd K_Nearest_Neighbors
```

2. Install dependencies:

```
pip install numpy pandas scikit-learn matplotlib
```

### Usage

1. Open and run evaluate.ipynb in Jupyter Notebook or JupyterLab.
2. Follow the notebook to load the Iris dataset, split data, and evaluate KNNClassifier.
3. Compare results with KNeighborsClassifier from scikit-learn.

## Motivation

The motivation behind this project is to gain a deeper understanding of the K-Nearest Neighbors algorithm by implementing it from scratch. While scikit-learn provides a robust and efficient implementation, building the algorithm from the ground up offers valuable insights into its mechanics and inner workings. This hands-on approach helps in:

- **Learning the fundamentals:** By coding the algorithm, one can understand the step-by-step process of how KNN works.
- **Enhancing problem-solving skills:** Tackling potential issues and debugging helps improve problem-solving abilities.
- **Appreciating existing libraries:** Understanding the complexities involved in creating machine learning algorithms increases appreciation for the tools provided by libraries like scikit-learn.
- **Educational purposes:** This project serves as an educational tool for others who wish to learn about KNN and its implementation.