import pandas as pd
import random

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.constants.random_data import data_dict


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Attempting to predict score based on user inputs")

            model_path = "artifacts/04_Models/best_model.pkl"
            preprc_path = "artifacts/03_Pipeline/ppln_prpc.pkl"

            model = load_object(file_path=model_path)
            preprc = load_object(file_path=preprc_path)

            x_pred_tf = preprc.transform(features)
            y_pred = model.predict(x_pred_tf)

            return y_pred

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)


class CustomData:
    def __init__(
        self,
        MSZoning: str = None,
        LotArea: int = None,
        LotShape: str = None,
        LandContour: str = None,
        Utilities: str = None,
        LotConfig: str = None,
        OverallQual: str = None,
        MasVnrType: str = None,
        TotalBsmtSF: str = None,
        LowQualFinSF: str = None,
        BsmtFullBath: str = None,
        FullBath: int = None,
        KitchenAbvGr: int = None,
        GarageCars: int = None,
        GarageArea: int = None,
        PoolArea: int = None,
        PoolQC: str = None,
        SaleType: str = None,
    ):
        self.MSZoning = str(MSZoning)
        self.LotArea = int(LotArea)
        self.LotShape = str(LotShape)
        self.LandContour = str(LandContour)
        self.Utilities = str(Utilities)
        self.LotConfig = str(LotConfig)
        self.OverallQual = str(OverallQual)
        self.MasVnrType = str(MasVnrType)
        self.TotalBsmtSF = str(TotalBsmtSF)
        self.LowQualFinSF = str(LowQualFinSF)
        self.BsmtFullBath = str(BsmtFullBath)
        self.FullBath = int(FullBath)
        self.KitchenAbvGr = int(KitchenAbvGr)
        self.GarageCars = int(GarageCars)
        self.GarageArea = int(GarageArea)
        self.PoolArea = int(PoolArea)
        self.PoolQC = str(PoolQC)
        self.SaleType = str(SaleType)

    def get_DataFrame(self):
        try:
            logging.info("Converting user inputs to DataFrame")
            # Getting select data from customer
            cust_data = {
                "MSZoning": [self.MSZoning],
                "LotArea": [self.LotArea],
                "LotShape": [self.LotShape],
                "LandContour": [self.LandContour],
                "Utilities": [self.Utilities],
                "LotConfig": [self.LotConfig],
                "OverallQual": [self.OverallQual],
                "MasVnrType": [self.MasVnrType],
                "TotalBsmtSF": [self.TotalBsmtSF],
                "LowQualFinSF": [self.LowQualFinSF],
                "BsmtFullBath": [self.BsmtFullBath],
                "FullBath": [self.FullBath],
                "KitchenAbvGr": [self.KitchenAbvGr],
                "GarageCars": [self.GarageCars],
                "GarageArea": [self.GarageArea],
                "PoolArea": [self.PoolArea],
                "PoolQC": [self.PoolQC],
                "SaleType": [self.SaleType],
            }
            # Getting remaining data chosen at random
            rand_data = {key: [random.choice(val)] for key, val in data_dict.items()}
            # Full data
            full_data_dict = cust_data | rand_data

            df_cust = pd.DataFrame(full_data_dict)
            logging.info("Successfully converted user inputs to DataFrame")
            return df_cust

        except Exception as e:
            logging.info(f"Error: {e}")
            raise CustomException(e)
