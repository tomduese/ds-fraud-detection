import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import pickle
import warnings

warnings.filterwarnings("ignore")

import models.eda as eda


# load train set
client = eda.load_data("data/train/client_train.csv")
invoice = eda.load_data("data/train/invoice_train.csv")
train = eda.feature_change(client, invoice)
X_train, X_test, y_train, y_test = eda.sampled_train_test_split(train)


## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Feature engineering on train")
X_train = eda.get_mean_consumption(X_train)
X_train = eda.get_historical_mean(X_train)

# model
print("Training a simple linear regression")
reg = LinearRegression().fit(X_train, y_train)
y_train_pred = reg.predict(X_train)
roc_auc_train = eda.roc_auc(y_train, y_train_pred)

# feature eng on test data
print("Feature engineering on test")
X_test = eda.get_mean_consumption(X_test)
X_test = eda.get_historical_mean(X_test)

y_test_pred = reg.predict(X_test)
roc_auc_test = mean_squared_error(y_test, y_test_pred)

print(f"AUC on train is: {roc_auc_train}")
print(f"AUC on test is: {roc_auc_test}")
print("this is obviously fishy")
# saving the model
print("Saving model in the model folder")
filename = "models/xgboost_model.sav"
pickle.dump(reg, open(filename, "wb"))
