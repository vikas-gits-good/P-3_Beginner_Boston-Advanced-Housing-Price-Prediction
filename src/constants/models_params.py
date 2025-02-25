from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor,
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    PassiveAggressiveRegressor,
)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, XGBRFRegressor

models_dict = {
    # Ensemble Models
    "RandomForestRegressor": RandomForestRegressor(n_jobs=-1, random_state=44),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        loss="log_loss", criterion="friedman_mse", random_state=44
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(),
    "BaggingRegressor": BaggingRegressor(),
    "AdaBoostRegressor": AdaBoostRegressor(random_state=44),
    "XGBRegressor": XGBRegressor(),
    "XGBRFRegressor": XGBRFRegressor(),
    # # Tree Models
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=44),
    "ExtraTreeRegressor": ExtraTreeRegressor(),
    # Neightbour Models
    "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1),
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor(),
    # Linear Models
    "LinearRegression": LinearRegression(random_state=44, n_jobs=-1),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "ElasticNet": ElasticNet(),
    # Neural Network
    "MLPRegressor": MLPRegressor(random_state=44, activation="relu"),
}

params_dict = {
    "RandomForestClassifier": {
        "n_estimators": [700, 800],
        "max_depth": [8, 10],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [2, 3],
        "max_leaf_nodes": [30, 35],
        "min_impurity_decrease": [0.001],
    },
    "GradientBoostingClassifier": {
        "learning_rate": [0.4, 0.5],
        "n_estimators": [250, 300],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [3, 4],
        "max_depth": [5, 6],
        "max_leaf_nodes": [5, 6],
    },
    "AdaBoostClassifier": {
        "n_estimators": [100, 150],
        "learning_rate": [2, 3],
    },
    "XGBClassifier": {
        # "": [],
    },
    "XGBRFClassifier": {
        #     "": [],
    },
    "DecisionTreeClassifier": {
        "max_depth": [8, 9],
        "min_samples_leaf": [13, 14],
        "max_leaf_nodes": [5, 6],
    },
    "KNeighborsClassifier": {"p": [1, 2], "n_neighbors": [4, 5]},
    "LogisticRegression": {
        "C": [2, 3],
    },
    "MLPClassifier": {
        "learning_rate_init": [0.004, 0.005],
    },
}
