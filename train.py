import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings("ignore")

import models.eda as eda


# load train and test set
print("loading datasets...")
client = eda.load_data("data/train/client_train.csv")
invoice = eda.load_data("data/train/invoice_train.csv")
train = eda.feature_change(client, invoice)
train, test = eda.sampling(train)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder...")
test.to_csv("data/test.csv", index=False)

# print("Feature engineering on train...")
# train = eda.feature_engineering(train)
X_train, y_train = eda.feature_target_split(train)

print("scaling train...")
scaler = StandardScaler()
X_train_preprocessed = scaler.fit_transform(X_train)

# model
print("Training a simple linear regression...")
logreg = LogisticRegression().fit(X_train_preprocessed, y_train)
y_train_pred = logreg.predict(X_train_preprocessed)
roc_auc_train = eda.roc_auc(y_train, y_train_pred)

print("Training a XGBoost model...")
# Define XGBoost model with enable_categorical=True and tree_method='hist'
xgb_model = XGBClassifier(
    objective="binary:logistic",
    seed=42,
    enable_categorical=True,
    tree_method="hist",
    n_estimators=400,
    max_depth=15,
    learning_rate=0.1,
).fit(X_train_preprocessed, y_train)
y_pred = xgb_model.predict(X_train_preprocessed)
roc_auc_train_xgb = eda.roc_auc(y_train, y_pred)

# feature engineering on test data
print("Feature engineering on test...")
# test = eda.feature_engineering(test)
X_test, y_test = eda.feature_target_split(test)
X_test_preprocessed = scaler.transform(X_test)

y_test_pred = logreg.predict(X_test_preprocessed)
roc_auc_test = eda.roc_auc(y_test, y_test_pred)

y_test_pred = xgb_model.predict(X_test_preprocessed)
roc_auc_test_xgb = eda.roc_auc(y_test, y_test_pred)

print(f"AUC on train with logistic regression is: {roc_auc_train.round(2)}")
print(f"AUC on train with XGB is: {roc_auc_train_xgb.round(2)}")
print("-" * 50)
print(f"AUC on test is: {roc_auc_test.round(2)}")
print(f"AUC on test with XGB is: {roc_auc_test_xgb.round(2)}")
print("this is obviously fishy")
# saving the model
print("Saving model in the model folder...")
filename = "models/logreg_model.sav"
pickle.dump(
    {"logreg": logreg, "xgb": xgb_model, "scaler": scaler}, open(filename, "wb")
)
print("done")
