from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import config


def create_pipeline():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("compress", PCA(n_components=0.95, random_state=config.RANDOM_STATE)),
            ("select_k_best", SelectKBest(k=100, score_func=mutual_info_classif)),
        ]
    )


mlp_params = {
    "activation": "relu",
    "solver": "lbfgs",
    "alpha": 9.846852933268213,
    "max_iter": 1000,
}
knc_param = {
    "n_neighbors": 4,
    "weights": "distance",
    "algorithm": "brute",
    "leaf_size": 11,
}
lr_param = {"C": 0.9383975744448804, "solver": "newton-cg", "max_iter": 1000}
rf_param = {
    "n_estimators": 36,
    "criterion": "entropy",
    "max_depth": 18,
    "min_samples_split": 8,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
}
dt_param = {
    "criterion": "log_loss",
    "max_depth": 15,
    "min_samples_split": 3,
    "min_samples_leaf": 6,
    "max_features": None,
}

mlp = MLPClassifier(**mlp_params)
knc = KNeighborsClassifier(**knc_param)
rf = RandomForestClassifier(**rf_param)
lr = LogisticRegression(**lr_param)
dt = DecisionTreeClassifier(**dt_param)
qda = QuadraticDiscriminantAnalysis()
