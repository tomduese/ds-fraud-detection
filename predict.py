import sys
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

import models.eda as eda

print("Number of arguments:", len(sys.argv), "arguments.")
print("Argument List:", str(sys.argv))

# in an ideal world this would validated
if len(sys.argv) > 1:
    model_path = sys.argv[1]
    test_path = sys.argv[2]
else:
    model_path = "models/logreg_model.sav"
    test_path = "data/test.csv"

# load the model from disk
loaded_data = pickle.load(open(model_path, "rb"))
logreg = loaded_data["logreg"]
xgb = loaded_data["xgb"]
scaler = loaded_data["scaler"]
test = pd.read_csv(test_path)

# feature eng on test data
print("Feature engineering")
test = eda.feature_engineering(test)
X_test, y_test = eda.feature_target_split(test)
X_test_preprocessed = scaler.transform(X_test)

# predict with the logreg model
print("Predicting with logistic regression on test:")
y_test_pred = logreg.predict(X_test_preprocessed)
eda.print_metrics(y_test, y_test_pred)

# predict with the logreg model
print("Predicting with XGB on test:")
y_test_pred = xgb.predict(X_test_preprocessed)
eda.print_metrics(y_test, y_test_pred)
