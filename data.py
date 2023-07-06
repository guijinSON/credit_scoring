import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_dataframe(df, numerical_columns, categorical_columns):
    """
    Preprocess a data frame by standardizing numerical columns and encoding categorical columns.
    
    :param df: input data frame
    :param numerical_columns: list of numerical column names
    :param categorical_columns: list of categorical column names
    :return: preprocessed data frame
    """
    
    # Make a copy of the input data frame to avoid modifying the original
    df_copy = df.copy()
    
    # Standardize numerical columns
    scaler = StandardScaler()
    df_copy[numerical_columns] = scaler.fit_transform(df_copy[numerical_columns].astype(float))
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col])
        label_encoders[col] = le

    # Return the preprocessed data frame and label encoders (in case you need them later)
    return df_copy, label_encoders
