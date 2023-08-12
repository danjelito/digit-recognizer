from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import config

pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        ("compress", PCA(n_components=0.95, random_state=config.RANDOM_STATE)),
        ("select_k_best", SelectKBest(k=100, score_func=mutual_info_classif)),
    ]
)

knc_param = {
    "n_neighbors": 4,
    "weights": "distance",
    "algorithm": "ball_tree",
    "leaf_size": 12,
}
lr_param = {"C": 2.2458127933062992, "solver": "newton-cg"}
rf_param = {
    "n_estimators": 109,
    "criterion": "gini",
    "max_depth": 60,
    "min_samples_split": 8,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

knc = KNeighborsClassifier(**knc_param)
rf = RandomForestClassifier(**rf_param)
lr = LogisticRegression(**lr_param)
