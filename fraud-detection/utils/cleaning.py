import pandas as pd
import numpy as np
import re

def parse_amount(value):
    if pd.isna(value):
        return np.nan
    
    value = str(value)
    value = re.sub(r'[^\d.]', '', value)

    try:
        return float(value)
    except:
        return np.nan


def parse_timestamp(value):
    try:
        return pd.to_datetime(value, errors='coerce')
    except:
        return pd.NaT


def normalize_location(loc):
    if pd.isna(loc):
        return "unknown"
    
    loc = loc.strip().lower()

    mapping = {
        "bombay": "mumbai",
        "mum": "mumbai",
        "mumbai": "mumbai"
    }

    return mapping.get(loc, loc)


def clean_data(df):

    # Merge amt column
    if 'amt' in df.columns:
        df['transaction_amount'] = df['transaction_amount'].fillna(df['amt'])

    df['transaction_amount'] = df['transaction_amount'].apply(parse_amount)
    df['transaction_timestamp'] = df['transaction_timestamp'].apply(parse_timestamp)

    df['user_location'] = df['user_location'].apply(normalize_location)
    df['merchant_location'] = df['merchant_location'].apply(normalize_location)

    df = df.drop_duplicates()

    return df

def data_quality_report(df):

    report = {}

    # Missing values
    report['missing_values'] = df.isnull().sum().to_dict()

    # Duplicate rows
    report['duplicate_rows'] = int(df.duplicated().sum())

    # Invalid IPs (simple check)
    def is_valid_ip(ip):
        if pd.isna(ip):
            return False
        
        parts = str(ip).split('.')
        if len(parts) != 4:
            return False
        
        for part in parts:
            if not part.isdigit():
                return False
            if int(part) < 0 or int(part) > 255:
                return False
        
        return True

    if 'ip_address' in df.columns:
        invalid_ips = df['ip_address'].apply(lambda x: not is_valid_ip(x))
        report['invalid_ip_count'] = int(invalid_ips.sum())
    else:
        report['invalid_ip_count'] = 0

    return report