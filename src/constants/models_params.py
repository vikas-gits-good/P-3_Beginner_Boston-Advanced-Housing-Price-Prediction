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
    "RandomForestRegressor": RandomForestRegressor(
        criterion="friedman_mse", n_jobs=-1, random_state=44
    ),
    "GradientBoostingRegressor": GradientBoostingRegressor(
        loss="squared_error", criterion="friedman_mse", random_state=44
    ),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        loss="squared_error", max_iter=300, random_state=44
    ),
    "BaggingRegressor": BaggingRegressor(random_state=44, n_jobs=-1),
    "AdaBoostRegressor": AdaBoostRegressor(loss="square", random_state=44),
    "XGBRegressor": XGBRegressor(),
    "XGBRFRegressor": XGBRFRegressor(),
    # # Tree Models
    "DecisionTreeRegressor": DecisionTreeRegressor(
        criterion="friedman_mse", random_state=44
    ),
    "ExtraTreeRegressor": ExtraTreeRegressor(criterion="friedman_mse", random_state=44),
    # Neightbour Models
    "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1),
    "RadiusNeighborsRegressor": RadiusNeighborsRegressor(n_jobs=-1),
    # Linear Models
    "LinearRegression": LinearRegression(n_jobs=-1),
    "PassiveAggressiveRegressor": PassiveAggressiveRegressor(random_state=44),
    "Lasso": Lasso(random_state=44),
    "Ridge": Ridge(random_state=44),
    "ElasticNet": ElasticNet(random_state=44),
    # Neural Network
    "MLPRegressor": MLPRegressor(random_state=44),
}

params_dict = {
    "RandomForestRegressor": {
        "n_estimators": [200, 400],
        "max_depth": [8, 10],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [2, 3],
        "max_leaf_nodes": [30, 35],
        "min_impurity_decrease": [0.001],
    },
    "GradientBoostingRegressor": {
        "learning_rate": [0.1, 0.4, 0.5],
        "n_estimators": [250, 300],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [3, 4],
        "max_depth": [5, 6],
        "max_leaf_nodes": [5, 6],
    },
    "HistGradientBoostingRegressor": {
        "learning_rate": [0.1, 0.4, 0.5],
        "min_samples_leaf": [3, 4],
        "max_depth": [5, 6],
        "max_leaf_nodes": [5, 6],
    },
    "BaggingRegressor": {
        "n_estimators": [250, 300],
        "max_features": [30, 40],
        "max_samples": [350, 400],
    },
    "AdaBoostRegressor": {
        "n_estimators": [200, 250],
        "learning_rate": [0.5, 1],
    },
    "XGBRegressor": {
        "lambda": [1, 5, 10],
        "alpha": [0, 5, 10],
    },
    "XGBRFRegressor": {
        "lambda": [1, 5, 10],
        "alpha": [0, 5, 10],
    },
    "DecisionTreeRegressor": {
        "max_depth": [10, 12],
        "min_samples_leaf": [10, 12],
        "max_leaf_nodes": [5, 6],
    },
    "ExtraTreeRegressor": {
        "max_depth": [6, 8],
        "min_samples_split": [2, 3],
        "min_samples_leaf": [1, 2],
        "max_leaf_nodes": [30, 40],
    },
    "KNeighborsRegressor": {"p": [1, 2], "n_neighbors": [2, 3]},
    "RadiusNeighborsRegressor": {
        # '':[],
    },
    "LinearRegression": {
        # "": [],
    },
    "PassiveAggressiveRegressor": {
        # "": [],
    },
    "Lasso": {
        "alpha": [1, 200],
    },
    "Ridge": {
        "alpha": [1, 6],
    },
    "ElasticNet": {
        "l1_ratio": [0.5, 0.8, 0.9],
    },
    "MLPRegressor": {
        # "": [],
    },
}
