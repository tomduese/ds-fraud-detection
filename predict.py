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
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, "rb"))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

# feature eng on test data
print("Feature engineering")
X_test = eda.get_mean_consumption(X_test)
X_test = eda.get_historical_mean(X_test)

# predict with the model
y_test_pred = loaded_model.predict(X_test)
mse_test = eda.print_metrics(y_test, y_test_pred)
