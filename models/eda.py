import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import random


def load_data(path):
    df = pd.read_csv(path)
    return df


def print_info(df):
    print("Cleaning DataFrame...")
    print("Printing Head of DataFrame:")
    print(df.head())

    print("Printing Info of DataFrame:")
    print(df.info())

    print("Printing Sum of Null Values in DataFrame:")
    print(df.isnull().sum())

    print("Loaded DataFrame with shape:", df.shape)


def feature_change(cl, inv):
    cl["client_catg"] = cl["client_catg"].astype("category")
    cl["disrict"] = cl["disrict"].astype("category")
    cl["region"] = cl["region"].astype("category")
    cl["region_group"] = cl["region"].apply(
        lambda x: 100 if x < 100 else 300 if x > 300 else 200
    )
    cl["creation_date"] = pd.to_datetime(cl["creation_date"])

    cl["coop_time"] = (2019 - cl["creation_date"].dt.year) * 12 - cl[
        "creation_date"
    ].dt.month

    drop_cols = [
        "consommation_level_1",
        "consommation_level_2",
        "consommation_level_3",
        "consommation_level_4",
        "old_index",
        "new_index",
        "invoice_date",
        "creation_date",
    ]
    inv["counter_type"] = inv["counter_type"].map({"ELEC": 1, "GAZ": 0})
    inv["counter_statue"] = inv["counter_statue"].map(
        {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            769: 5,
            "0": 0,
            "5": 5,
            "1": 1,
            "4": 4,
            "A": 0,
            618: 5,
            269375: 5,
            46: 5,
            420: 5,
        }
    )
    inv["invoice_date"] = pd.to_datetime(inv["invoice_date"], dayfirst=True)
    inv["invoice_month"] = inv["invoice_date"].dt.month
    inv["invoice_year"] = inv["invoice_date"].dt.year
    inv["is_weekday"] = (
        (pd.DatetimeIndex(inv.invoice_date).dayofweek) // 5 == 1
    ).astype(float)
    inv["total_consumption"] = inv["new_index"] - inv["old_index"]
    cl.rename(columns={"disrict": "district"}, inplace=True)

    # df_client and df_invoice are being merged on the client_id column
    df = pd.merge(cl, inv, on="client_id", how="left")
    df["client_id"] = df["client_id"].str.split("_").str[-1].astype(int)
    df.drop(drop_cols, inplace=True, axis=1)
    return df


def get_mean_consumption(df):
    # Calculate the mean consumption per year for each client
    mean_consumption = (
        df.groupby(["client_id", "invoice_year"])["total_consumption"]
        .mean()
        .reset_index()
    )

    # Merge the mean consumption dataframe back into the original dfframe
    df = df.merge(mean_consumption, on=["client_id", "invoice_year"], how="left")
    df.rename(
        columns={
            "total_consumption_y": "mean_consumption_per_year",
            "total_consumption_x": "total_consumption",
        },
        inplace=True,
    )
    return df


# Define a function to calculate the historical mean consumption for each client
def historical_mean(group):
    # Sort the group by invoice year in ascending order
    group = group.sort_values("invoice_year")

    # Calculate the expanding mean for the total consumption column
    group["historical_mean_consumption"] = group["total_consumption"].expanding().mean()

    return group


def get_historical_mean(df):
    # Apply the function to the dataframe grouped by client id
    df = df.groupby("client_id").apply(historical_mean)
    df = df.reset_index(drop=True)
    # Shift the historical mean consumption column by one row to avoid using the current year's mean
    df["historical_mean_consumption"] = df.groupby("client_id")[
        "historical_mean_consumption"
    ].shift()
    df.historical_mean_consumption = df.historical_mean_consumption.fillna(0)
    return df


def print_metrics(y_test, y_pred):
    print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
    print("Precision: {:.2f}".format(precision_score(y_test, y_pred)))
    print("Recall: {:.2f}".format(recall_score(y_test, y_pred)))
    print("F1 Score: {:.2f}".format(f1_score(y_test, y_pred)))
    print("ROC AUC Score: {:.2f}".format(roc_auc_score(y_test, y_pred)))


def roc_auc(y_test, y_pred):
    return roc_auc_score(y_test, y_pred)


def sampled_train_test_split(df):
    random.seed(42)
    honest = df[df.target == 0]
    fraud = df[df.target == 1]

    honest_id = honest.client_id.unique().tolist()
    fraud_id = fraud.client_id.unique().tolist()

    train_number_of_clients = int(len(fraud_id) * 0.7)
    test_number_of_clients = len(fraud_id) - train_number_of_clients

    train_honest = random.sample(honest_id, k=train_number_of_clients)
    train_fraud = random.sample(fraud_id, k=train_number_of_clients)

    # Get all the items that were not selected in the random sample
    remaining_honest = [item for item in honest_id if item not in train_honest]
    test_honest = random.sample(remaining_honest, k=test_number_of_clients)
    test_fraud = [item for item in fraud_id if item not in train_fraud]

    train_ids = train_honest + train_fraud
    test_ids = test_honest + test_fraud

    # create train and test datasets
    df_train = df[df.client_id.isin(train_ids)]
    df_test = df[df.client_id.isin(test_ids)]

    X_train = df_train.drop("target", axis=1)
    X_test = df_test.drop("target", axis=1)
    y_train = df_train.target
    y_test = df_test.target

    return df_train, df_test
