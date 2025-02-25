import pandas as pd
import numpy as np
from typing import Literal

from src.logger import logging
from src.exception import CustomException

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, MinMaxScaler

from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class Save_DataFrame(BaseEstimator, TransformerMixin):
    """Save Dataframe generated during preprocessing and feature engineeering phase to .pkl for analysis

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def fit(self, X, y=None):
        try:
            self.column_names = list(X.columns)
            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def transform(self, X, y=None):
        try:
            if X.shape[0] == 1:
                if X.shape[1] > 78:  # Double check this
                    self.save_path = "artifacts/02_DataFrames/Predict/df_pred_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Predict/df_pred_fs.pkl"

            elif X.shape[0] > 300:
                if X.shape[1] > 78:
                    self.save_path = "artifacts/02_DataFrames/Train/df_train_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Train/df_train_fs.pkl"

            elif X.shape[0] < 300:
                if X.shape[1] > 78:
                    self.save_path = "artifacts/02_DataFrames/Train/df_test_pp.pkl"
                else:
                    self.save_path = "artifacts/02_DataFrames/Train/df_test_fs.pkl"

            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.column_names)
            if self.save_path:
                X.to_pickle(self.save_path)
                logging.info(f"DataFrame saved to {self.save_path}")
            return X

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.column_names


class RareItemReplacer(BaseEstimator, TransformerMixin):
    """Custom class to replace rate categorical labels with fill_value

    Args:
        BaseEstimator (object): Pipeline compatibility
        TransformerMixin (object): pipeline compatibility
    """

    def __init__(self, fill_value: str = "rare_item", target: str = "SalePrice"):
        self.fill_value = fill_value
        self.target = target
        self.col_catg_dict = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.cols_list = X.columns.tolist()
        if y is not None:
            df = df.join(pd.DataFrame(data=y, columns=[self.target]))
        for col in self.cols_list:
            temp = df.groupby(col)[self.target].count() / len(df)
            self.col_catg_dict[col] = temp[temp > 0.01].index.tolist()
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.cols_list:
            df[col] = np.where(
                df[col].isin(self.col_catg_dict[col]), df[col], self.fill_value
            )
        return df

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols_list


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Custom class to encode categorical labels based on its influence on target variable

    Args:
        BaseEstimator (object): Pipeline compatibility
        TransformerMixin (object): Pipeline compatibility
    """

    def __init__(self, target: str = "SalePrice"):
        self.target = target
        self.label_ord_dict = {}

    def fit(self, X, y=None):
        df = X.copy()
        self.cols_list = X.columns.tolist()
        if y is not None:
            df = df.join(pd.DataFrame(data=y, columns=[self.target]))
        for col in self.cols_list:
            labels_ordered = (
                df.groupby(col)[self.target].mean().sort_values().index.tolist()
            )
            labels_ordered = {k: i for i, k in enumerate(labels_ordered, 0)}
            self.label_ord_dict[col] = labels_ordered
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.cols_list:
            df[col] = df[col].map(self.label_ord_dict[col])
        return df

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.cols_list


class MultiModelEstimator(BaseEstimator, TransformerMixin):
    """Pipeline compatible method to train multiple models

    Args:
        BaseEstimator (object): For pipeline compatibility
        TransformerMixin (object): For pipeline compatibility
    """

    def __init__(
        self,
        models: dict[dict],
        param_grids: dict[dict],
        cv=3,
        scoring: str = None,
        method: Literal[
            "GridSearchCV", "RandomizedSearchCV", "Optuna"
        ] = "GridSearchCV",
    ):
        self.models = models
        self.param_grids = param_grids
        self.cv = cv
        self.scoring = scoring
        self.method = method
        self.grid_searches = {}

        if not set(models.keys()).issubset(set(param_grids.keys())):
            missing_params = list(set(models.keys()) - set(param_grids.keys()))
            logging.info("They keys in model dict isnt matching that in params dict")
            raise ValueError(
                "Some estimators are missing parameters: %s" % missing_params
            )

    def fit(self, X, y=None):
        """Iterates over each model looking for the best hyperparametes

        Args:
            X (array/DataFrame): independant feature
            y (array, optional): dependant feature. Kept for consistency. Defaults to None.

        Raises:
            CustomException: Error during hyperparameter tuning

        Returns:
            self: calculates best paramters and stores it for use with predict method
        """
        try:
            for name, model in self.models.items():
                logging.info(f"Fitting {self.method} for {name}")

                if self.method == "GridSearchCV":
                    gs = GridSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "RandomizedSearchCV":
                    gs = RandomizedSearchCV(
                        model,
                        self.param_grids[name],
                        cv=self.cv,
                        scoring=self.scoring,
                        refit=True,
                        n_jobs=-1,
                    )
                    gs.fit(X, y)

                elif self.method == "Optuna":
                    pass

                self.grid_searches[name] = gs
                logging.info(f"Best parameters for {name}: {gs.best_params_}")

            return self

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def predict(self, X):
        """Iterates over each model and predicts the dependant variable

        Args:
            X (array/DataFrame): independant features from the test/valdiation set

        Raises:
            CustomException: Error during prediction

        Returns:
            Tuple[DataFrame,dict(Models)]: DataFrame with y_pred from each model and a dictionary of models
        """
        try:
            self.predictions_ = {}
            self.models_ = {}

            for name, grid_search in self.grid_searches.items():
                logging.info(f"Predicting {self.method} for {name}")
                best_model = grid_search.best_estimator_
                # if hasattr(best_model, "predict_proba"):
                #     self.predictions_[name] = best_model.predict_proba(X)
                #     self.models_[name] = best_model
                # else:
                #     self.predictions_[name] = best_model.predict(X)
                #     self.models_[name] = best_model
                self.predictions_[name] = best_model.predict(X)
                self.models_[name] = best_model
            df_y_pred = pd.DataFrame(self.predictions_)
            return df_y_pred, self.models_

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.predictions_.keys())


class PipelineConstructor:
    """Class that creates a pipeline that will do preprocessing and feature engineering and cant be fit with the model training pipeline"""

    def __init__(
        self,
        cols_numr_cont: list[str] = None,
        cols_catg: list[str] = None,
        cols_drop: list[str] = None,
    ):
        self.cols_numr_cont = cols_numr_cont
        self.cols_catg = cols_catg
        self.cols_drop = cols_drop

    def create_new_feats(
        self, data: pd.DataFrame = None, drop: list[str] = None
    ) -> pd.DataFrame:
        try:
            df = data.copy()
            # Create Year Based Feature
            df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
            df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
            df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
            # Drop unnecessary colummns
            if drop:
                df.drop(columns=drop, inplace=True)

            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def CappingOutlier(
        self,
        data,
        columns: list[str] = None,
        threshold: float = 3,
        method: Literal["z_score", "iqr"] = "z_score",
    ):
        try:
            # Temporary workaraound. The dropped columns sent here through self.vols_numr is causing KeyError
            columns = [col for col in data.columns if data[col].dtypes != "O"]
            df = data.copy()
            for col in columns:
                if method == "z_score":
                    mean = df[col].mean()
                    std = df[col].std()
                    upper_bound = mean + threshold * std
                    lower_bound = mean - threshold * std
                elif method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + threshold * IQR
                    lower_bound = Q1 - threshold * IQR

                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            return df

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)

    def create_pipeline(self):
        """Creates a preprocessing + feature engineering pipeline.
        Method is already defined.

        Raises:
            CustomException: Error in pipeline creation

        Returns:
            object: pipeline object to be used with model training
        """
        try:
            ppln = Pipeline(
                steps=[
                    (
                        "Preprocess",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "Numerical_Feat_1",
                                    Pipeline(
                                        steps=[
                                            ("Imputer", KNNImputer(n_neighbors=5)),
                                            (
                                                "Outlier",
                                                FunctionTransformer(
                                                    func=self.CappingOutlier,
                                                    kw_args={
                                                        "threshold": 3,
                                                        "method": "z_score",
                                                    },
                                                ),
                                            ),
                                            (
                                                "Transformer",
                                                PowerTransformer(
                                                    method="yeo-johnson",
                                                    standardize=False,
                                                ),
                                            ),
                                            (
                                                "Scaler",
                                                MinMaxScaler(feature_range=(0, 1)),
                                            ),
                                        ]
                                    ),
                                    self.cols_numr_cont,
                                ),
                                (
                                    "Categorical_Feat",
                                    Pipeline(
                                        steps=[
                                            (
                                                "Imputer",
                                                SimpleImputer(
                                                    strategy="constant",
                                                    fill_value="Missing",
                                                ),
                                            ),
                                            (
                                                "Rare_labels",
                                                RareItemReplacer(
                                                    fill_value="rare_item",
                                                    target="SalePrice",
                                                ),
                                            ),
                                            (
                                                "Encoder",
                                                CategoricalEncoder(target="SalePrice"),
                                            ),
                                        ]
                                    ),
                                    self.cols_catg,
                                ),
                            ],
                            remainder="passthrough",
                            verbose_feature_names_out=False,
                            force_int_remainder_cols=False,
                        ),
                    ),
                    (
                        "New_Feats",
                        FunctionTransformer(
                            func=self.create_new_feats, kw_args={"drop": self.cols_drop}
                        ),
                    ),
                    ("Save_DF_prpc", Save_DataFrame()),
                    (
                        "Feat_Slcn",
                        RFECV(
                            estimator=Ridge(alpha=75),
                            min_features_to_select=1,
                            cv=StratifiedKFold(n_splits=5),
                            scoring="r2",
                            n_jobs=-1,
                        ),
                    ),
                    ("Save_DF_ftsl", Save_DataFrame()),
                ]
            ).set_output(transform="pandas")

            return ppln

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
