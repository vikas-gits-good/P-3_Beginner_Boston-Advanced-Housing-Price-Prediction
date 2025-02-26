from flask import Flask, request, render_template

from src.logger_predict import logging
from src.pipeline.pipeline_prediction import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        logging.info("Getting user input from web form")
        data = CustomData(
            MSZoning=request.form.get("MSZoning"),
            LotArea=request.form.get("LotArea"),
            LotShape=request.form.get("LotShape"),
            LandContour=request.form.get("LandContour"),
            Utilities=request.form.get("Utilities"),
            LotConfig=request.form.get("LotConfig"),
            OverallQual=request.form.get("OverallQual"),
            MasVnrType=request.form.get("MasVnrType"),
            TotalBsmtSF=request.form.get("TotalBsmtSF"),
            LowQualFinSF=request.form.get("LowQualFinSF"),
            BsmtFullBath=request.form.get("BsmtFullBath"),
            FullBath=request.form.get("FullBath"),
            KitchenAbvGr=request.form.get("KitchenAbvGr"),
            GarageCars=request.form.get("GarageCars"),
            GarageArea=request.form.get("GarageArea"),
            PoolArea=request.form.get("PoolArea"),
            PoolQC=request.form.get("PoolQC"),
            SaleType=request.form.get("SaleType"),
        )
        x_pred = data.get_DataFrame()

        logging.info("Calculating prediction for user input")
        ppln_pred = PredictionPipeline()
        y_pred = ppln_pred.predict(features=x_pred)
        logging.info("Returning prediction to user")
        return render_template("home.html", results=round(y_pred[0], 2))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
