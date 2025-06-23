import pandas as pd
import pdfplumber

def read_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            first_page = pdf.pages[0]  # Extract tables from the first page
            table = first_page.extract_table()
            if table:
                return pd.DataFrame(table[1:], columns=table[0])  # Convert to a DataFrame
    return None  # Return None if the file format is unsupported

# Llamaindex maybe a helpful tool to connect to various data sources like API, PDFs, databases, and more. 
