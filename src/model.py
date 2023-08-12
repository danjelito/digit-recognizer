from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import config

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("compress", PCA(n_components=0.95, random_state=config.RANDOM_STATE)),
    ("select_k_best", SelectKBest(k=100, score_func=mutual_info_classif)),
])

knc_param = {'n_neighbors': 5, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 35}
# lr_param = 
# rf_param = 

knc = KNeighborsClassifier(**knc_param)
# rf = RandomForestClassifier(**rf_param)
# lr = LogisticRegression(**lr_param)
