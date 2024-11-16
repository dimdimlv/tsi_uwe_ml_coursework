# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)


def manage_outliers(df):
    numerical_columns = ['Credit_Amount', 'Bill_Amount_09', 'Bill_Amount_08',
                         'Bill_Amount_07', 'Bill_Amount_06', 'Bill_Amount_05',
                         'Bill_Amount_04', 'Pay_Amount_09', 'Pay_Amount_08',
                         'Pay_Amount_07', 'Pay_Amount_06', 'Pay_Amount_05', 'Pay_Amount_04']
    # Skip Age as it does not need outlier management

    for col in numerical_columns:
        cap_outliers(df, col)

    for col in numerical_columns:
        df[col] = df[col].apply(lambda x: np.log1p(x) if x > 0 else 0)  # handle zero or negative values

    return df


def encode_categorical_variables(df):
    # Define the ordinal mapping for Pay_Status_* variables
    ordinal_mapping = {
        -2: 0,  # No consumption (lowest severity)
        -1: 1,  # Paid in full
        0: 2,   # Paid on time
        1: 3,   # 1-month delay
        2: 4,   # 2-month delay
        3: 5,   # 3-month delay
        4: 6,   # 4-month delay
        5: 7,   # 5-month delay
        6: 8,   # 6-month delay
        7: 9,   # 7-month delay
        8: 10,  # 8-month delay
        9: 11,  # 9-month delay or more (highest severity)
    }

    # Define the categorical columns
    pay_status_columns = ['Pay_Status_09', 'Pay_Status_08', 'Pay_Status_07',
                          'Pay_Status_06', 'Pay_Status_05', 'Pay_Status_04']
    nominal_columns = ['Education', 'Gender', 'Marital_Status']
    # Assuming 'Default' is already encoded as 0/1 and doesn't need transformation

    # Apply ordinal encoding for Pay_Status_* variables
    for col in pay_status_columns:
        if col in df.columns:
            df[col] = df[col].map(ordinal_mapping).fillna(0).astype('int64')

    # Apply one-hot encoding for nominal variables
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

    # Convert one-hot encoded columns to int64
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype('int64')

    # Convert 'Default' column to int64
    df['Default'] = df['Default'].astype('int64')

    return df


def scale_numerical_columns(df, numerical_columns):
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df