import pandas as pd
import numpy as np
import streamlit as st
from components.LinkDB import DataFromDatalab

import pandas as pd
import numpy as np

def get_csv_metadata(df):
    """
    Processes uploaded DataFrame to extract:
    - Cleaned datetime column
    - Valid numeric columns
    - Text-based filters (with unique values)
    - Metadata display table

    Returns:
        df (pd.DataFrame): cleaned DataFrame
        metadata_df (pd.DataFrame): unique values of text columns
        value_col_names (list): numeric column names available for selection
    """
    try:
        # Normalize column names for robustness
        df.columns = df.columns.str.strip()

        # === 1. Ensure PERIOD_DT column exists ===
        if "PERIOD_DT" not in df.columns:
            st.markdown("🚨 **Your level of one data needs to have `PERIOD_DT` in columns.**")
            return pd.DataFrame(), pd.DataFrame(), []  # or raise an Exception, if you prefer

        df["PERIOD_DT"] = pd.to_datetime(df["PERIOD_DT"], errors="coerce")

        df_cleaned = df.copy()  # <-- preserve original values

        # === 2. Detect numeric columns (coerce as needed) ===
        value_col_names = []
        for col in df.columns:
            if col == "PERIOD_DT":
                continue
            num_unique = df[col].nunique(dropna=True)
            try:
                coerced = pd.to_numeric(df[col], errors="coerce")
                valid_ratio = coerced.notnull().sum() / max(1, df[col].notnull().sum())

                # Only treat as numeric if it's not low-cardinality (like a text category)
                if valid_ratio >= 0.9 and num_unique > 15:  # Adjust threshold as needed
                    df_cleaned[col] = coerced
                    value_col_names.append(col)
            except Exception:
                continue  # skip if conversion fails

        # === 3. Detect text columns (after coercion) ===
        text_columns = df.select_dtypes(include=["object"]).copy()
        filtered_text = text_columns.drop(columns=[col for col in value_col_names if col in text_columns.columns], errors="ignore")

        # Safeguard: remove any unnamed or empty-string columns
        valid_text_cols = [col for col in filtered_text.columns if col.strip() != "" and col in df.columns]
        #filtered_text = filtered_text[valid_text_cols]

        # === 4. Collect unique values per text column ===
        unique_values = {
            col: filtered_text[col].dropna().unique().tolist()
            for col in valid_text_cols
        }

        # Pad for visual alignment
        max_len = max((len(vals) for vals in unique_values.values()), default=0)
        padded_data = {
            col: vals + [np.nan] * (max_len - len(vals))
            for col, vals in unique_values.items()
        }

        metadata_df = pd.DataFrame(padded_data)
        column_names = list(unique_values.keys())
        metadata_df.insert(0, "Column Name", column_names + [np.nan] * (max_len - len(column_names)))
        metadata_df.insert(1, "Data Type", [str(df[col].dtype) for col in column_names] + [np.nan] * (max_len - len(column_names)))


        return df_cleaned, metadata_df, value_col_names

    except Exception as e:
        # Fallback in case something serious breaks
        return pd.DataFrame(), pd.DataFrame(), []


def get_csv_metadata_old(df):
    """
    Extracts metadata from a CSV file:
    - Lists column names.
    - Displays unique values for text-based columns.
    - Ignores numeric and date columns.
    
    :param file_path: Path to the CSV file.
    :return: Pandas DataFrame with column names and unique values.
    """
    try:
        # Load CSV into a DataFrame
        df = df

        # Convert PERIOD_dT to datetime format
        if "PERIOD_DT" in df.columns:
            df["PERIOD_DT"] = pd.to_datetime(df["PERIOD_DT"], errors="coerce")  # Handle errors gracefully
        else: st.markdown("🚨 **Your level of one data needs to have `PERIOD_DT` in columns.**")

        # Convert VOLUME to numeric format
        value_col = df.select_dtypes(include=["int64","float64"]).columns.tolist()
        value_col_value = df.select_dtypes(include=["int64","float64"]).columns
        for col in value_col_value.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Handle non-numeric values gracefully, bad formats -> NaN

        # Identify text-based columns (excluding numerical & date columns)
        text_columns = df.select_dtypes(include=["object"])  # 'object' covers string columns
        
        # **Exclude PERIOD_dT and VOLUME from metadata**
        excluded_columns = ["PERIOD_DT"] + value_col
        text_columns = text_columns.drop(columns=[col for col in excluded_columns if col in text_columns.columns], errors="ignore")

        # Get unique values for each text column
        unique_values = {col: text_columns[col].dropna().unique().tolist() for col in text_columns.columns}

        # Find the max list length
        max_length = max(len(values) for values in unique_values.values()) if unique_values else 0

        # Pad shorter lists with NaN
        padded_values = {col: values + [np.nan] * (max_length - len(values)) for col, values in unique_values.items()}

        # Convert metadata dictionary into a DataFrame
        metadata_df = pd.DataFrame(padded_values)

        # Insert column names
        metadata_df.insert(0, "Column Name", list(unique_values.keys()) + [""] * (max_length - len(unique_values)))

        # Insert data types for each column
        metadata_df.insert(1, "Data Type", [str(df[col].dtype) for col in unique_values.keys()] + [""] * (max_length - len(unique_values)))

        return df, metadata_df, value_col # Return both the processed DataFrame & metadata

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})  , pd.DataFrame(), []

#example to use:
# db = DataFromDatalab()  
# df = db.DEI_bell()
# get_csv_metadata(df)
