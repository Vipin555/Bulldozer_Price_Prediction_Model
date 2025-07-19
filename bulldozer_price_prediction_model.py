# -*- coding: utf-8 -*-
"""bulldozer_price_prediction_model.ipynb

Original file is located at
    https://colab.research.google.com/drive/12cR0DcuTYLrmrwBlqxAtOT_0DvOdBNYz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("bluebook-for-bulldozers/bluebook-for-bulldozers/TrainAndValid.csv",low_memory = False)

df.info()

df.isna().sum()

df.head()

df.SalePrice.plot.hist()

df = pd.read_csv("bluebook-for-bulldozers/bluebook-for-bulldozers/TrainAndValid.csv",
                low_memory= False,
                parse_dates = ["saledate"])

df.saledate.dtype

df.saledate[:1000]

fig , ax = plt.subplots()
ax.scatter(df["saledate"][:1000],df["SalePrice"][:1000])

df.sort_values(by=["saledate"], inplace=True,ascending=True)
df.saledate.head(10)

df_tmp = df.copy()
df_tmp.head()

df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayofWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayofYear"] = df_tmp.saledate.dt.dayofyear

df_tmp.head().T

df_tmp.drop("saledate", axis=1,inplace=True)

df_tmp.state.value_counts()

# This will identify all columns of dtype 'object' or 'string'
for label, content in df_tmp.items():
    if content.dtype == 'object' or pd.api.types.is_string_dtype(content):
        print(label)

for label , content in df_tmp.items():
    if content.dtype == "object" or pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

df_tmp.info()

df_tmp.to_csv("bluebook-for-bulldozers/bluebook-for-bulldozers/Train_tmp.csv",index=False)

df_tmp = pd.read_csv("bluebook-for-bulldozers/bluebook-for-bulldozers/Train_tmp.csv",low_memory=False)

for label , content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

for label , content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

for label,content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            df_tmp[label] = content.fillna(content.median())

df_tmp.head()

for label , content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        df_tmp[label] = pd.Categorical(content).codes+1

df_tmp.isna().sum()

df_tmp.drop("auctioneerID_is_missing",axis=1,inplace=True)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_jobs=-1,random_state=42,max_samples=10000)

df_tmp.saleYear

df_tmp.saleYear.value_counts()

df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

X_train , y_train = df_train.drop("SalePrice",axis=1), df_train.SalePrice
X_valid , y_valid = df_val.drop("SalePrice", axis=1), df_val.SalePrice

from sklearn.metrics import mean_squared_log_error , mean_absolute_error , r2_score

def rmsle(y_test,y_preds):
    return np.sqrt(mean_squared_log_error(y_test,y_preds))

def show_score(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Training MAE": mean_absolute_error(y_train,train_preds),
              "Valid MAE": mean_absolute_error(y_valid,val_preds),
              "Training RMSLE": rmsle(y_train,train_preds),
              "Valid RMSLE": rmsle(y_valid,val_preds),
              "Training R^2": r2_score(y_train,train_preds),
              "Valid R^2": r2_score(y_valid,val_preds)
    }
    return scores

model.fit(X_train,y_train)

show_score(model)

"""## HyperParameter Tuning Using RandomizedSearchCV

"""

from sklearn.model_selection import RandomizedSearchCV

rf_grid = {"n_estimators":np.arange(10,100,10),
          "max_depth": [None,3,5,10],
          "min_samples_split":np.arange(2,20,2),
          "min_samples_leaf":np.arange(1,20,2),
          "max_features":[0.5,1,"sqrt","log2"],
          "max_samples":[10000]}

rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                   random_state=42),
                                                   param_distributions=rf_grid,
                                                   n_iter=2,
                                                   cv=5)

rs_model.fit(X_train,y_train)

show_score(rs_model)

"""### Best Paramwters training achived after 100 iterations"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# ideal_model = RandomForestRegressor(n_estimators=40,
#                                    min_samples_leaf=1,
#                                    max_features=0.5,
#                                    min_samples_split=14,
#                                    n_jobs=-1,
#                                    max_samples=None,
#                                    random_state = 42)
# ideal_model.fit(X_train,y_train)

show_score(ideal_model)

df_test = pd.read_csv("bluebook-for-bulldozers/bluebook-for-bulldozers/Test.csv",
                       low_memory=False,
                       parse_dates=["saledate"])
df_test.head()

"""### Preprocessing the data"""

for label , content in df_test.items():
    if content.dtype == "object" or pd.api.types.is_string_dtype(content):
        df_test[label] = content.astype("category").cat.as_ordered()

def preprocess_data(df):
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayofWeek"] = df.saledate.dt.dayofweek
    df["saleDayofYear"] = df.saledate.dt.dayofyear

    df.drop("saledate",axis=1,inplace=True)

    for label , content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label+"_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        if not pd.api.types.is_numeric_dtype(content):
            df[label+"_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes+1


    return df

df_test = preprocess_data(df_test)
df_test.head()

X_train.head()

set(X_train.columns) - set(df_test.columns)

test_preds = ideal_model.predict(df_test)

df_preds = pd.DataFrame()
df_preds["SalesID"] = df_test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds

df_preds.to_csv("bluebook-for-bulldozers/test_prediction.csv" , index = False)

"""### Feature Importance"""

def plot_features(columns,importances,n=20):
     df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
          .sort_values("feature_importances", ascending=False)
          .reset_index(drop=True))

     fig , ax = plt.subplots()
     ax.barh(df["features"][:n],df["feature_importances"][:20])
     ax.set_ylabel("Features")
     ax.set_xlabel("Feature importance")
     ax.invert_yaxis()

plot_features(X_train.columns , ideal_model.feature_importances_)

