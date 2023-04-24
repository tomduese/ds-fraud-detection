import pandas as pd


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
    inv.drop(drop_cols, inplace=True, axis=1)
    cl.rename(columns={"disrict": "district"}, inplace=True)

    # df_client and df_invoice are being merged on the client_id column
    df = pd.merge(cl, inv, on="client_id", how="left")
    df["client_id"] = df["client_id"].str.slice(start=13).astype(int)

    return df
