import pandas as pd

df = pd.read_csv("artifacts/01_Data/train.csv").drop(columns=["SalePrice"])

cols_requ = [
    "MSZoning",
    "LotArea",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "OverallQual",
    "MasVnrType",
    "TotalBsmtSF",
    "LowQualFinSF",
    "BsmtFullBath",
    # "BsmtHalfBath",
    "FullBath",
    # "HalfBath",
    "KitchenAbvGr",
    "GarageCars",
    "GarageArea",
    # "WoodDeckSF",
    "PoolArea",
    "PoolQC",
    "SaleType",
]

cols_rand = [col for col in df.columns if col not in cols_requ]
data_dict = {col: df[col].unique().tolist() for col in cols_rand}
