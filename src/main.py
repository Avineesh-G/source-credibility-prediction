"""
Implements various traditional machine learning classifiers
for source credibility / fake news detection.
"""
from src.evaluation import CredibilityEvaluator
from src.data_preprocessing import DataPreprocessor


import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TraditionalMLModels:
    def __init__(self, model_type="logistic_regression", random_state=42):
        """
        model_type: one of
        ["logistic_regression", "naive_bayes", "random_forest",
         "decision_tree", "svm", "gradient_boost", "adaboost"]
        """
        self.model_type = model_type
        self.model = None
        self.random_state = random_state
        self._init_model()

    def _init_model(self):
        if self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=self.random_state, max_iter=500)
        elif self.model_type == "naive_bayes":
            self.model = MultinomialNB()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=self.random_state, n_estimators=200)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        elif self.model_type == "svm":
            self.model = SVC(kernel="linear", probability=True, random_state=self.random_state)
        elif self.model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(random_state=self.random_state)
        elif self.model_type == "adaboost":
            self.model = AdaBoostClassifier(random_state=self.random_state, n_estimators=100)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train, tune_hyperparameters=False, param_grid=None, cv_folds=3):
        if tune_hyperparameters and param_grid:
            grid = GridSearchCV(self.model, param_grid, cv=cv_folds, n_jobs=-1, scoring="accuracy")
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"[INFO] Best Parameters: {grid.best_params_}")
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("This model does not support probability prediction")

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[INFO] Model saved to {path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[INFO] Model loaded from {path}")
