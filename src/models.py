# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def get_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=42)


def get_random_forest():
    return RandomForestClassifier(random_state=42)


def get_svm():
    return SVC(kernel='linear', probability=True, random_state=42)


def get_knn():
    return KNeighborsClassifier(n_neighbors=5)


def get_naive_bayes():
    return GaussianNB()


def get_xgboost():
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method="gpu_hist", random_state=42)
