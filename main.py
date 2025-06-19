import pandas as pd

import apps.app
import argparse

from predictors.complete.complete_regressor import CompleteRegressor

from quantile_forest import RandomForestQuantileRegressor  # Assuming this is how it's imported

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import feature_selection

from drivers import base as driver
from apps import app
from helpers import pickler

# helper
# [NOTE]: do not use X.loc for some reason it will take 100x longer to train
#         idk if it doesn't actually parse it correctly or what, but watch out
def generate_preprocessor(X):
    numeric_features = []
    categorical_features = []
    
    for col in X:
        # Identify what type each column is.
        isNumeric = True
        for rowIndex, row in X[col].items():
            try:
                # If it can be a float, make it a float.
                X[col][rowIndex] = float(X[col][rowIndex])
                # If the float is NaN (unacceptable to Sci-kit), make it -1.0 for now.
                if pd.isnull(X[col][rowIndex]):
                    X[col][rowIndex] = -1.0
            except:
                # Otherwise, we will assume this is categorical data.
                isNumeric = False
        if isNumeric:
            # For whatever reason, float conversions don't want to work in Pandas dataframes.
            # Try changing the value column-wide instead.
            # TODO: Doesn't seem to actually solve anything.
            X[col] = X[col].astype(float)
            numeric_features.append(str(col))
        else:
            categorical_features.append(str(col))

    # Standardization for numeric data.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())])

    # One-hot encoding for categorical data.
    categorical_transformer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    # Add the transformers to a preprocessor object.
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),])

    return preprocessor

def generate_empire_preprocessor(X):
    sel = feature_selection.VarianceThreshold(threshold=0)
    # X = sel.fit_transform(X)
    # # Discretization. Buckets results to whole minutes like related works.
    # # y = y.apply(lambda x: int(x/60))

    # Track the feature types of each cell.
    numeric_features = []
    categorical_features = []
    # Iterate over every cell.
    for col in X:
        try:
            # Attempt to convert column to float; if successful, it's numeric.
            X[col] = pd.to_numeric(X[col], errors='raise')
            X[col] = X[col].astype(float)
            X[col].fillna(-1.0, inplace=True)
            numeric_features.append(str(col))
        except ValueError:
            # Conversion failed, so treat as categorical and convert to string.
            X[col] = X[col].astype(str)
            categorical_features.append(str(col))

    # Standardization for numeric data.
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())])
    # One-hot encoding for categorical data.
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    # Add the transformers to a preprocessor object.
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),],
        sparse_threshold=0.0)

    return preprocessor

def get_pipeline(preprocessor, clf):
    """ 
    Convenience function to add a preprocessor to a regression pipeline.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def example_run_of_EMPIRE():
    assert True==False, "NOT IMPLEMENTED FILE"

    emp = app.EMPIRE("timeTaken","datapath.csv")
    X,y = emp.parse()
    preprocessor = generate_empire_preprocessor(X)
    

    driver.Base().run(
        get_pipeline(preprocessor,RandomForestRegressor()),
                      "random forest regressor on empire",
                      X,y)



if __name__ == "__main__":
    print("hello world!")

    # getting args
    parser = argparse.ArgumentParser(
        description="Test and train a predictive framework."
    )
    parser.add_argument("--depickle", action='store_true',
                        help="Whether to compute ML models or depickle already-made ones from file")
    args = parser.parse_args()

    # setting pickler
    pickler.set_should_depickle(args.depickle)

    # setting apps
    apps = [
        app.Nekbone("timeTaken","./tests/nekbonedataset.csv"),
        # app.HACC_IO("timeTaken","./tests/HACC-IOdataset.csv"),
        # app.SWFFT("timeTaken","./tests/SWFFTdataset.csv"),
        # app.ExaMiniMD("timeTaken","./tests/ExaMiniMDsnapdataset.csv"),
    ]

    # test
    x = apps[0]
    # print(x.generate_test(x.get_params(),1))
    # print(x.get_params())

    for app in apps:
        print("running app: ",app.name)
        X,y = app.parse()
        preprocessor = generate_preprocessor(X)
        # driver.Base().run(get_pipeline(preprocessor,RandomForestRegressor()),"Random Forest Regressor "+app.name,X,y)
        driver.Quantile().run(get_pipeline(preprocessor, RandomForestQuantileRegressor()),
                    "Quantile Forest "+app.name,
                    X,y,[0.5,0.75,0.95,0.975,0.985,0.99,0.995,0.999])

    #
    # # X,y = v.parse()
    # # preprocessor = generate_preprocessor(X)
    # # driver.Base().run(get_pipeline(preprocessor,RandomForestRegressor()),"Random Forest Regressor "+v.name,X,y)
    # # driver.Single().run(CompleteRegressor(list(X.columns)),"complete regressor"+v.name,X,y)
    # # driver.Quantile().run(get_pipeline(preprocessor, RandomForestQuantileRegressor()),
    # #                 "Quantile Forest "+v.name,
    # #                 X,y,[0.5,0.75,0.95,0.975,0.985,0.99,0.995,0.999])
