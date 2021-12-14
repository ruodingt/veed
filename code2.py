# randomly restricted allowable imports
import numpy as np
from typing import Tuple
from collections import Counter

# Feel free to use the OOTB impelmentation for the base DecisionTree
from sklearn.tree import DecisionTreeClassifier

# import for performance comparisons
from sklearn.ensemble import RandomForestClassifier

# data to test on
from sklearn import datasets
from sklearn.model_selection import train_test_split
#
# import warnings
#
# warnings.filterwarnings('ignore')


def bootstrap_sample(X: np.array, y: np.array) -> Tuple[np.array, np.array]:
    """
    Stub for the bootstrapping method
    TODO: numba

    This should bootstrap the existing data (sampling with replacement)
    and return the resampled datasets (X_samp, y_samp)
    """
    indices = y.shape[0]
    resampled_indices = np.random.choice(a=range(indices), size=indices, replace=True)
    X_resampled = X[resampled_indices]
    y_resampled = y[resampled_indices]
    return X_resampled, y_resampled


class BasicRandomForestClassifier:
    def __init__(self, n_trees=11,
                 min_samples_split=3,
                 max_depth=100,
                 n_feats=None):
        # set up some hyperparameters for our classifier
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        # storage container for the tree ensemble
        self.trees = []

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Stub for the fit method of the RF class

        This method fits a set of DecisionTreeClassifiers
        and adds the result back to the self.trees
        class variable.

        Each tree is fit on a bootstrapped sample of the
        original training dataset X and target labels y.
        """
        for _ in range(self.n_trees):
            X_resampled, y_resampled = bootstrap_sample(X, y)
            _t = DecisionTreeClassifier(
                min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            _t.fit(X_resampled, y_resampled)
            self.trees.append(_t)

    def predict(self, X: np.array) -> np.array:
        """
        Stub for the prediction method of the RF class

        This method uses the ensemble of decision trees stored in
        self.trees to vote on the most likely class for each data sample.

        Here we'll implement the "most common class" voting system -
        the class that occurs most frequently should be selected. If there's a tie,
        we randomly sample the winner.
        """
        _predicted = []
        for t in self.trees:
            _p = t.predict(X)
            _predicted.append(_p)
            # counts = np.bincount(a)
            # print(np.argmax(counts))
        # _predicted_ensemble = [Counter(p).most_common(1)[0][0] for p in np.stack(_predicted).T.tolist()]
        # _predicted_ensemble = [np.argmax(np.bincount(p)) for p in np.stack(_predicted).T.tolist()]
        p = np.stack(_predicted).T.max(axis=1)
        return p


# Testing
if __name__ == "__main__":
    # Boilerplate to train the classifier on a toy dataset

    # simple acc
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


    # load data
    data = datasets.load_iris()
    X = np.concatenate([data.data] * 1000)
    y = np.concatenate([data.target] * 1000, )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # fit and run the custom classifier we just build
    bc_clf = BasicRandomForestClassifier()

    bc_clf.fit(X_train, y_train)
    y_pred = bc_clf.predict(X_test)
    basic_acc = accuracy(y_test, y_pred)

    print("Custom Accuracy:", basic_acc)

    # compare to the sklearn implementation
    clf = RandomForestClassifier()

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    base_acc = accuracy(y_test, y_pred)

    print("Sklearn Accuracy:", base_acc)
    print("How well did we do? Base minus our custom accuracy (diff): ", - base_acc + basic_acc)
