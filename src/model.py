from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import config

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("compress", PCA(n_components=0.95, random_state=config.RANDOM_STATE)),
    ("select_k_best", SelectKBest(k=100, score_func=mutual_info_classif)),
])

param= {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 35}
knc= KNeighborsClassifier(**param)