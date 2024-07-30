import sys
import os

# Calculate the project root directory
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "SU-EFG")
)

if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import numpy as np
from fitter import Fitter, get_common_distributions
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.model_selection import train_test_split
import joblib
from utilities.utils import log_transform

warnings.filterwarnings("ignore")

pd.set_option("future.no_silent_downcasting", True)


def _initialization():
    orders_df = pd.read_csv("../../Data/orders_data_competition.csv")
    clients_df = pd.read_csv("../../Data/clients_data_competition.csv")

    orders_df.dropna(inplace=True)
    orders_df = orders_df[orders_df["Order Via"] == "Online"]
    orders_df.drop(columns=["Order Via", "Security ID"], inplace=True)

    account_ids_to_remove = clients_df[
        (clients_df["Client Type Name"] != "Individuals")
        | (clients_df["Company Name"] != "HSB")
    ]["Account ID"].unique()

    # Remove the accounts from the accounts dataframe where Client Type Name is not 'individuals'
    clients_df = clients_df[~clients_df["Account ID"].isin(account_ids_to_remove)]

    # Remove the orders of these accounts from the orders dataframe
    orders_df = orders_df[~orders_df["Account ID"].isin(account_ids_to_remove)]

    clients_df.drop(columns=["Client Type Name", "Company Name"], inplace=True)

    return orders_df, clients_df


def _get_num_of_orders_account_level(orders_df: pd.DataFrame):
    num_of_orders = orders_df.groupby("Account ID").size()
    num_of_orders = num_of_orders.fillna(0)
    return num_of_orders.astype(int)


def _get_first_order_date_account_level(orders_df: pd.DataFrame):
    first_order_date = orders_df.groupby("Account ID")["Order Time"].min()
    return pd.to_datetime(first_order_date, format="%d-%m-%Y")


def _get_last_order_date_account_level(orders_df: pd.DataFrame):
    last_order_date = orders_df.groupby("Account ID")["Order Time"].max()
    return pd.to_datetime(last_order_date, format="%d-%m-%Y")


def _get_num_of_completed_orders(orders_df: pd.DataFrame):
    num_of_completed_orders = orders_df.groupby("Account ID")["Is Completed"].sum()
    num_of_completed_orders = num_of_completed_orders.fillna(0)
    return num_of_completed_orders.astype(int)


def _get_num_of_canceled_orders(orders_df: pd.DataFrame):
    num_of_canceled_orders = orders_df.groupby("Account ID")["Is Canceled"].sum()
    num_of_canceled_orders = num_of_canceled_orders.fillna(0)
    return num_of_canceled_orders.astype(int)


def _get_avg_price_account_level(orders_df: pd.DataFrame):
    avg_price = (
        orders_df.groupby("Account ID")["Price"].sum()
        / orders_df.groupby("Account ID").size()
    )
    return avg_price.fillna(0.0)


def _get_total_quantity_orderd_account_level(orders_df: pd.DataFrame):
    total_quantity = orders_df.groupby("Account ID")["Quantity"].sum()
    total_quantity = total_quantity.fillna(0)
    return total_quantity.astype(int)


def _get_most_frequent_value(
    column_name: str, agg_orders: pd.DataFrame, orders_df: pd.DataFrame
):
    temp = pd.DataFrame(data=agg_orders.iloc[:, 0])
    col_name = f"Most_Frequent_{column_name}"
    temp[col_name] = orders_df.groupby("Account ID")[column_name].apply(
        lambda x: x.mode().iloc[0]
    )
    return pd.concat([agg_orders, temp.iloc[:, 1:]], axis=1)


def _calculate_age(birth_date):
    # Assuming birth_date is a datetime object
    today = datetime.today()
    return (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )


def _aggregate_orders_data(orders_df: pd.DataFrame, clients_df: pd.DataFrame):
    agg_orders = pd.DataFrame()

    agg_orders["AccountID"] = clients_df["Account ID"].unique()

    agg_orders["NumOfOrders"] = _get_num_of_orders_account_level(orders_df=orders_df)

    # Convert the 'order_date' column to datetime
    orders_df["Order Time"] = pd.to_datetime(orders_df["Order Time"])

    agg_orders["FirstOrder"] = _get_first_order_date_account_level(orders_df=orders_df)

    agg_orders["LastOrder"] = _get_last_order_date_account_level(orders_df=orders_df)

    agg_orders["NumOfCompleted"] = _get_num_of_completed_orders(orders_df=orders_df)

    agg_orders["NumOfCanceled"] = _get_num_of_canceled_orders(orders_df=orders_df)

    agg_orders["AvgPrice"] = _get_avg_price_account_level(orders_df=orders_df)

    agg_orders["TotalQuantity"] = _get_total_quantity_orderd_account_level(
        orders_df=orders_df
    )

    agg_orders = _get_most_frequent_value(
        "Order Type", agg_orders=agg_orders, orders_df=orders_df
    )

    agg_orders = _get_most_frequent_value(
        "Execution Status", agg_orders=agg_orders, orders_df=orders_df
    )

    agg_orders = _get_most_frequent_value(
        "Sector Name", agg_orders=agg_orders, orders_df=orders_df
    )

    return agg_orders


def _clean_and_process_clients_data(clients_df: pd.DataFrame):
    clients_df.dropna(inplace=True)

    clients_df = clients_df[clients_df["Is Client Suspended"] == 0]

    clients_df.drop(columns="Is Client Suspended", inplace=True)

    clients_df["OpenDate"] = pd.to_datetime(clients_df["OpenDate"], format="%m/%d/%Y")

    clients_df["BirthDate"] = pd.to_datetime(clients_df["BirthDate"], format="%Y-%m-%d")

    clients_df["Age"] = clients_df["BirthDate"].apply(_calculate_age)

    clients_df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)

    return clients_df


def _merge_orders_with_clients(
    agg_orders: pd.DataFrame, clients_df: pd.DataFrame, orders_df: pd.DataFrame
):
    df_account_level = pd.merge(clients_df, agg_orders, on="AccountID", how="inner")

    # Assume now is the current datetime
    now = datetime.now()

    # Calculate the midpoint dates for all accounts
    df_account_level["MidpointDate"] = (
        df_account_level["OpenDate"] + (now - df_account_level["OpenDate"]) / 2
    )

    # Calculate the days open until midpoint and from midpoint to now
    df_account_level["DaysOpenMidpoint"] = (
        df_account_level["MidpointDate"] - df_account_level["OpenDate"]
    ).dt.days
    df_account_level["DaysMidpointNow"] = (
        now - df_account_level["MidpointDate"]
    ).dt.days

    # Filter orders for each account and time period
    orders_df["Order Time"] = pd.to_datetime(orders_df["Order Time"])

    # Merge orders with accounts to avoid looping
    merged_df = orders_df.merge(
        df_account_level[["AccountID", "OpenDate", "MidpointDate"]],
        left_on="Account ID",
        right_on="AccountID",
    )

    # Calculate start and end period masks
    start_period_mask = (merged_df["Order Time"] >= merged_df["OpenDate"]) & (
        merged_df["Order Time"] < merged_df["MidpointDate"]
    )
    end_period_mask = (merged_df["Order Time"] >= merged_df["MidpointDate"]) & (
        merged_df["Order Time"] < now
    )

    # Calculate order and quantity sums for start and end periods
    start_period_df = merged_df[start_period_mask]
    end_period_df = merged_df[end_period_mask]

    # Group by AccountID to aggregate counts and sums
    start_grouped = (
        start_period_df.groupby("AccountID")
        .agg({"Order ID": "count", "Quantity": "sum"})
        .rename(
            columns={"Order ID": "NumOrdersStart", "Quantity": "QuantityOrderedStart"}
        )
    )
    end_grouped = (
        end_period_df.groupby("AccountID")
        .agg({"Order ID": "count", "Quantity": "sum"})
        .rename(columns={"Order ID": "NumOrdersEnd", "Quantity": "QuantityOrderedEnd"})
    )

    # Merge the aggregated data back to the account level dataframe
    df_account_level = df_account_level.merge(
        start_grouped, left_on="AccountID", right_index=True, how="left"
    ).merge(end_grouped, left_on="AccountID", right_index=True, how="left")

    # Fill NaN values with 0 (for accounts with no orders in either period)
    df_account_level.fillna(
        {
            "NumOrdersStart": 0,
            "QuantityOrderedStart": 0,
            "NumOrdersEnd": 0,
            "QuantityOrderedEnd": 0,
        },
        inplace=True,
    )

    # Calculate rates
    df_account_level["OrderRate_Start"] = (
        df_account_level["NumOrdersStart"] / df_account_level["DaysOpenMidpoint"]
    )
    df_account_level["OrderRate_End"] = (
        df_account_level["NumOrdersEnd"] / df_account_level["DaysMidpointNow"]
    )
    df_account_level["QuantityOrderedRate_Start"] = (
        df_account_level["QuantityOrderedStart"] / df_account_level["DaysOpenMidpoint"]
    )
    df_account_level["QuantityOrderedRate_End"] = (
        df_account_level["QuantityOrderedEnd"] / df_account_level["DaysMidpointNow"]
    )

    # Drop intermediate columns if not needed
    df_account_level.drop(
        columns=[
            "MidpointDate",
            "DaysOpenMidpoint",
            "DaysMidpointNow",
            "NumOrdersStart",
            "QuantityOrderedStart",
            "NumOrdersEnd",
            "QuantityOrderedEnd",
        ],
        inplace=True,
    )

    df_account_level.rename(columns=lambda x: x.replace(" ", ""), inplace=True)

    return df_account_level


def _aggregate_to_account_level(orders_df: pd.DataFrame, clients_df: pd.DataFrame):
    agg_orders = _aggregate_orders_data(orders_df=orders_df, clients_df=clients_df)
    clients_df = _clean_and_process_clients_data(clients_df=clients_df)
    df_account_level = _merge_orders_with_clients(
        agg_orders=agg_orders, clients_df=clients_df, orders_df=orders_df
    )
    return df_account_level


def _check_dormant(date):
    one_year_before_now = datetime.now() - timedelta(days=183)

    if date < one_year_before_now:
        return 1
    else:
        return 0


def _aggregate_existing_numerical_features(
    df_account_level: pd.DataFrame, df_client_level: pd.DataFrame
):
    df_client_level["Gender"] = (
        df_account_level.groupby("ClientID")["Gender"].first().reset_index()
    )["Gender"]

    df_client_level["Age"] = (
        df_account_level.groupby("ClientID")["Age"].first().reset_index()
    )["Age"]

    df_client_level["RiskRate"] = (
        df_account_level.groupby("ClientID")["RiskRate"].first().reset_index()
    )["RiskRate"]

    df_client_level["IsMale"] = df_client_level.apply(
        lambda row: (1 if row["Gender"] == "Male" else 0), axis=1
    )

    df_client_level["NumOfAccounts"] = (
        df_account_level.groupby("ClientID").size().values
    )

    df_client_level["NumOfClosedAccounts"] = (
        df_account_level.groupby("ClientID")["IsClosed"].sum().values
    )

    df_client_level["NumOfSuspendedAccounts"] = (
        df_account_level.groupby("ClientID")["IsProfileSuspended"].sum().values
    )

    df_client_level["NumOfOrders"] = (
        df_account_level.groupby("ClientID")["NumOfOrders"].sum().values
    )

    df_client_level["TotalQuantity"] = (
        df_account_level.groupby("ClientID")["TotalQuantity"].sum().values
    )

    df_client_level["AvgPrice"] = (
        df_account_level.groupby("ClientID")["AvgPrice"].sum()
        / df_account_level.groupby("ClientID").size()
    ).values

    df_client_level = df_client_level[df_client_level["NumOfOrders"] > 0].reset_index()

    return df_client_level


def _compute_ratios(df_account_level: pd.DataFrame, df_client_level: pd.DataFrame):
    df_client_level["NumOfCompletedOrders"] = (
        df_account_level.groupby("ClientID")["NumOfCompleted"].sum().values
    )

    df_client_level["NumOfCanceledOrders"] = (
        df_account_level.groupby("ClientID")["NumOfCanceled"].sum().values
    )

    df_client_level["CompletedOrdersRatio"] = df_client_level.apply(
        lambda row: (
            1
            if row["NumOfOrders"] == 0
            else row["NumOfCompletedOrders"] / row["NumOfOrders"]
        ),
        axis=1,
    )

    df_client_level["CanceledOrdersRatio"] = df_client_level.apply(
        lambda row: (
            1
            if row["NumOfOrders"] == 0
            else row["NumOfCanceledOrders"] / row["NumOfOrders"]
        ),
        axis=1,
    )

    df_client_level["ClosedAccountsRatio"] = (
        df_client_level["NumOfClosedAccounts"] / df_client_level["NumOfAccounts"]
    )

    df_client_level["SuspendedAccountsRatio"] = (
        df_client_level["NumOfSuspendedAccounts"] / df_client_level["NumOfAccounts"]
    )

    df_client_level["ActiveAccountsRatio"] = df_client_level.apply(
        lambda row: (
            0
            if (row["NumOfSuspendedAccounts"] + row["NumOfClosedAccounts"])
            >= row["NumOfAccounts"]
            else 1 - (row["ClosedAccountsRatio"] + row["SuspendedAccountsRatio"])
        ),
        axis=1,
    )

    return df_client_level


def _get_most_frequent_values_client_level(
    df_account_level: pd.DataFrame, df_client_level: pd.DataFrame
):
    grouped_df = (
        df_account_level.groupby("ClientID")["Most_Frequent_OrderType"]
        .apply(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    df_client_level = pd.concat([df_client_level, grouped_df.iloc[:, 1:]], axis=1)

    grouped_df = (
        df_account_level.groupby("ClientID")["Most_Frequent_ExecutionStatus"]
        .apply(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    df_client_level = pd.concat([df_client_level, grouped_df.iloc[:, 1:]], axis=1)

    grouped_df = (
        df_account_level.groupby("ClientID")["Most_Frequent_SectorName"]
        .apply(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    df_client_level = pd.concat([df_client_level, grouped_df.iloc[:, 1:]], axis=1)

    return df_client_level


def _compute_is_dormant_client_col(df_account_level: pd.DataFrame):
    last_order_date_across_accounts = pd.Series(
        df_account_level.groupby("ClientID")["LastOrder"].max().values
    )

    last_order_date_across_accounts = pd.to_datetime(
        last_order_date_across_accounts.dt.date
    )

    return last_order_date_across_accounts.apply(_check_dormant)


def _compute_rate_differences(
    df_account_level: pd.DataFrame, df_client_level: pd.DataFrame
):
    df_client_level["AvgOrderRate_Start"] = (
        df_account_level.groupby("ClientID")["OrderRate_Start"].mean().values
    )

    df_client_level["AvgOrderRate_End"] = (
        df_account_level.groupby("ClientID")["OrderRate_End"].mean().values
    )

    df_client_level["AvgOrderRate_Difference"] = (
        df_client_level["AvgOrderRate_End"] - df_client_level["AvgOrderRate_Start"]
    )

    df_client_level["AvgQuantityOrderedRate_Start"] = (
        df_account_level.groupby("ClientID")["QuantityOrderedRate_Start"].mean().values
    )

    df_client_level["AvgQuantityOrderedRate_End"] = (
        df_account_level.groupby("ClientID")["QuantityOrderedRate_End"].mean().values
    )

    df_client_level["AvgQuantityOrderedRate_Difference"] = (
        df_client_level["AvgQuantityOrderedRate_End"]
        - df_client_level["AvgQuantityOrderedRate_Start"]
    )

    return df_client_level


def _aggregate_to_client_level(df_account_level: pd.DataFrame):
    df_client_level = pd.DataFrame()

    df_client_level = pd.DataFrame(
        {
            "ClientID": df_account_level.groupby("ClientID")["AccountID"]
            .nunique()
            .index,
        }
    )

    df_client_level = _aggregate_existing_numerical_features(
        df_account_level=df_account_level, df_client_level=df_client_level
    )

    valid_client_ids = set(df_client_level["ClientID"])

    df_account_level = df_account_level[
        df_account_level["ClientID"].isin(valid_client_ids)
    ]

    df_client_level = _compute_ratios(
        df_account_level=df_account_level, df_client_level=df_client_level
    )

    df_client_level = _get_most_frequent_values_client_level(
        df_account_level=df_account_level, df_client_level=df_client_level
    )

    df_client_level["IsDormant"] = _compute_is_dormant_client_col(
        df_account_level=df_account_level
    )

    df_client_level = _compute_rate_differences(
        df_account_level=df_account_level, df_client_level=df_client_level
    )

    return df_client_level


def _bin_AvgOrderRate_Difference(value):
    if value < 0:
        return "Decreased"
    elif value == 0:
        return "Constant"
    else:
        return "Increased"


def _bin_AvgQuantityOrderedRate_Difference(value):
    if value < 0:
        return "Decreased"
    elif value == 0:
        return "Constant"
    else:
        return "Increased"


def _bin_CompletedOrdersRatio(value):
    if value == 0:
        return "None"
    elif value > 0 and value < 0.5:
        return "Less Than Half"
    elif value >= 0.5 and value < 1:
        return "More Than Half"
    else:
        return "All"


def _bin_CanceledOrdersRatio(value):
    if value == 0:
        return "None"
    elif 0 < value <= 0.1:
        return "Little"
    elif 0.1 < value <= 0.3:
        return "Moderate"
    elif 0.3 < value < 1:
        return "Most"
    elif value == 1:
        return "All"


def _bin_columns(df_client_level: pd.DataFrame):
    df_client_level["AvgOrderRate_Difference"] = df_client_level[
        "AvgOrderRate_Difference"
    ].apply(_bin_AvgOrderRate_Difference)

    df_client_level["AvgQuantityOrderedRate_Difference"] = df_client_level[
        "AvgQuantityOrderedRate_Difference"
    ].apply(_bin_AvgQuantityOrderedRate_Difference)

    df_client_level["CompletedOrdersRatio"] = df_client_level[
        "CompletedOrdersRatio"
    ].apply(_bin_CompletedOrdersRatio)

    df_client_level["CanceledOrdersRatio"] = df_client_level[
        "CanceledOrdersRatio"
    ].apply(_bin_CanceledOrdersRatio)

    return df_client_level


def _define_label(df_client_level: pd.DataFrame):
    df_client_level["Churned"] = (df_client_level["IsDormant"] == 1) | (
        df_client_level["ActiveAccountsRatio"] < 0.5
    )

    df_client_level = df_client_level.replace({True: 1, False: 0})

    df_client_level["Churned"] = df_client_level["Churned"].astype(int)

    return df_client_level


def _drop_unnecessary_columns(df_client_level: pd.DataFrame):
    return df_client_level.drop(
        columns=[
            "IsDormant",
            "ActiveAccountsRatio",
            "SuspendedAccountsRatio",
            "TotalQuantity",
            "AvgOrderRate_Start",
            "AvgOrderRate_End",
            "AvgQuantityOrderedRate_Start",
            "AvgQuantityOrderedRate_End",
            "NumOfClosedAccounts",
            "NumOfAccounts",
            "NumOfSuspendedAccounts",
            "NumOfOrders",
            "NumOfCompletedOrders",
            "NumOfCanceledOrders",
            "ClosedAccountsRatio",
            "Gender",
            "index",
        ],
    )


def _split_train_test(df_client_level: pd.DataFrame):
    train_df, test_df = train_test_split(
        df_client_level, test_size=0.2, random_state=42
    )

    return train_df, test_df


def _one_hot_encoder(
    categorical_columns: list[str], train_df: pd.DataFrame, test_df: pd.DataFrame
):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[categorical_columns])

    train_encoded = encoder.transform(train_df[categorical_columns])
    test_encoded = encoder.transform(test_df[categorical_columns])

    train_encoded_df = pd.DataFrame(
        train_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )
    test_encoded_df = pd.DataFrame(
        test_encoded, columns=encoder.get_feature_names_out(categorical_columns)
    )

    train_encoded_df = train_encoded_df.astype("int8")
    test_encoded_df = test_encoded_df.astype("int8")

    train_df = train_df.drop(categorical_columns, axis=1).reset_index(drop=True)
    test_df = test_df.drop(categorical_columns, axis=1).reset_index(drop=True)

    train_df = pd.concat([train_df, train_encoded_df], axis=1)
    test_df = pd.concat([test_df, test_encoded_df], axis=1)

    return train_df, test_df, encoder


def _OHE(train_df: pd.DataFrame, test_df: pd.DataFrame):
    columns = [
        "RiskRate",
        "AvgOrderRate_Difference",
        "AvgQuantityOrderedRate_Difference",
        "CompletedOrdersRatio",
        "CanceledOrdersRatio",
        "Most_Frequent_OrderType",
        "Most_Frequent_ExecutionStatus",
        "Most_Frequent_SectorName",
    ]

    train_df, test_df, encoder = _one_hot_encoder(
        categorical_columns=columns, train_df=train_df, test_df=test_df
    )

    train_df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    test_df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)

    return train_df, test_df, encoder


def _get_normalized_data(data, dist):
    if dist == "uniform":
        scaler = MinMaxScaler().fit(data)
    elif dist == "norm":
        scaler = StandardScaler().fit(data)
    else:
        return log_transform
    return scaler


def _get_best_distribution(columns: list[str], df: pd.DataFrame):
    columns_distributions_dict = {column: "" for column in columns}

    for column in columns:
        print("###### " + column + " ######")
        data = df[column].values

        f = Fitter(
            data,
            distributions=get_common_distributions(),
        )
        f.fit()
        f.summary(plot=False)
        dist = f.get_best(method="sumsquare_error")
        best_dist = list(dist.keys())[0]

        columns_distributions_dict[column] = str(best_dist)
        print(column)
        print(f"Best Distribution: {best_dist}")
        print()

    return columns_distributions_dict


def _normalize(train_df: pd.DataFrame, test_df: pd.DataFrame):
    columns = [
        "Age",
        "AvgPrice",
    ]

    columns_distributions_dict = _get_best_distribution(columns, train_df)
    scalers = {}

    for column, dist in columns_distributions_dict.items():
        train_data = np.array(train_df[column]).reshape(-1, 1)
        test_data = np.array(test_df[column]).reshape(-1, 1)
        scaler = _get_normalized_data(train_data, dist)

        if dist in ["uniform", "norm"]:
            train_df[column] = scaler.transform(train_data)
            test_df[column] = scaler.transform(test_data)
            scalers[column] = scaler
        else:
            train_df[column] = scaler(train_data)
            test_df[column] = scaler(test_data)
            scalers[column] = scaler

    return train_df, test_df, scalers


def _saving(
    train_df: pd.DataFrame, test_df: pd.DataFrame, encoder: OneHotEncoder, scalers: dict
):
    cols = list(train_df.columns)
    cols.append(cols.pop(cols.index("Churned")))
    train_df = train_df[cols]

    cols = list(test_df.columns)
    cols.append(cols.pop(cols.index("Churned")))
    test_df = test_df[cols]

    train_df = train_df.iloc[:, 1:]
    test_df = test_df.iloc[:, 1:]

    columns = [
        "Churned",
        "IsMale",
    ]

    for col in columns:
        train_df[col] = train_df[col].astype("int8")
        test_df[col] = test_df[col].astype("int8")

    train_df.to_csv("../../Data/train_set.csv", index=False)

    test_df.to_csv("../../Data/test_set.csv", index=False)

    with open("../../pickle_files/encoder.pkl", "wb") as f:
        joblib.dump(encoder, f)

    with open("../../pickle_files/scalers.pkl", "wb") as f:
        joblib.dump(scalers, f)

    return train_df, test_df


def preprocess_data():
    print("##### Initializing #####")
    orders_df, clients_df = _initialization()
    print("----- Finished initialization -----\n")

    print("##### Aggregating to account level #####")
    df_account_level = _aggregate_to_account_level(
        orders_df=orders_df, clients_df=clients_df
    )
    print("----- Finished aggregating to account level -----\n")

    print("##### Aggregating to client level #####")
    df_client_level = _aggregate_to_client_level(df_account_level=df_account_level)
    print("----- Finished aggregating to client level -----\n")

    print("##### Binning columns #####")
    df_client_level = _bin_columns(df_client_level=df_client_level)
    print("----- Finished binning columns -----\n")

    print("##### Defining the label #####")
    df_client_level = _define_label(df_client_level=df_client_level)
    print("----- Finished defining the label -----\n")

    print("##### Dropping unnecessary columns #####")
    df_client_level = _drop_unnecessary_columns(df_client_level=df_client_level)
    print("----- Finished dropping unnecessary columns -----\n")

    print("##### Splitting train and test sets #####")
    train_df, test_df = _split_train_test(df_client_level=df_client_level)
    print("----- Finished splitting train and test sets -----\n")

    print("##### OHE #####")
    train_df, test_df, encoder = _OHE(train_df=train_df, test_df=test_df)
    print("----- Finished OHE -----\n")

    print("##### Normalizing #####")
    train_df, test_df, scalers = _normalize(train_df=train_df, test_df=test_df)
    print("----- Finished normalizing -----\n")

    print("##### Saving #####")
    train_df, test_df = _saving(
        train_df=train_df, test_df=test_df, encoder=encoder, scalers=scalers
    )
    print("----- Saved -----\n")

    return train_df, test_df
