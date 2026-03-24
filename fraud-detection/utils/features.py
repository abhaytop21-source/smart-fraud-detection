import pandas as pd

def create_features(df):

    df = df.sort_values(by=['user_id', 'transaction_timestamp'])

    # Avg amount per user
    df['avg_amount'] = df.groupby('user_id')['transaction_amount'].transform('mean')

    # Deviation
    df['amount_deviation'] = df['transaction_amount'] / (df['avg_amount'] + 1)

    # New device
    df['device_count'] = df.groupby(['user_id', 'device_id']).cumcount()
    df['is_new_device'] = (df['device_count'] == 0).astype(int)

    # New location
    df['location_count'] = df.groupby(['user_id', 'user_location']).cumcount()
    df['is_new_location'] = (df['location_count'] == 0).astype(int)

    # Time features
    df['hour'] = df['transaction_timestamp'].dt.hour
    df['is_night'] = df['hour'].apply(lambda x: 1 if x < 6 or x > 22 else 0)

    return df

def generate_explanations(df):

    explanations = []

    for i, row in df.iterrows():
        reasons = []

        if row['amount_deviation'] > 3:
            reasons.append("High amount deviation")

        if row['is_new_device'] == 1:
            reasons.append("New device used")

        if row['is_new_location'] == 1:
            reasons.append("New location")

        if row['is_night'] == 1:
            reasons.append("Unusual time (night)")

        if len(reasons) == 0:
            explanations.append("Normal transaction")
        else:
            explanations.append(", ".join(reasons))

    return explanations

def calculate_risk_score(df):

    scores = []
    levels = []

    for i, row in df.iterrows():
        score = 0

        if row['amount_deviation'] > 3:
            score += 40
        elif row['amount_deviation'] > 2:
            score += 25

        if row['is_new_device'] == 1:
            score += 20

        if row['is_new_location'] == 1:
            score += 20

        if row['is_night'] == 1:
            score += 20

        score = min(score, 100)

        scores.append(score)

        # Risk Level
        if score < 30:
            levels.append("Low")
        elif score < 70:
            levels.append("Medium")
        else:
            levels.append("High")

    # ✅ VERY IMPORTANT LINE
    return scores, levels

def detect_fraud_type(df):

    fraud_types = []

    for i, row in df.iterrows():
        types = []

        if row['amount_deviation'] > 3:
            types.append("High Amount")

        if row['is_new_device'] == 1:
            types.append("New Device")

        if row['is_new_location'] == 1:
            types.append("New Location")

        if row['is_night'] == 1:
            types.append("Night Activity")

        if len(types) == 0:
            fraud_types.append("Normal")
        else:
            fraud_types.append(", ".join(types))

    return fraud_types

def add_anomaly_features(df):

    # 🔹 Amount anomaly (very high values)
    df['amount_anomaly'] = df['transaction_amount'] > df['transaction_amount'].quantile(0.99)

    # 🔹 Invalid IP
    df['invalid_ip'] = df['ip_address'].astype(str).str.contains("not_an_ip|nan|999")

    # 🔹 Missing payment method
    df['missing_payment'] = df['payment_method'].isna() | (df['payment_method'] == "")

    # 🔹 Invalid location (weird characters)
    df['invalid_location'] = df['merchant_location'].astype(str).str.contains("#")

    # 🔹 Low balance anomaly
    df['low_balance'] = df['account_balance'] <= 0

    # 🔹 Device anomaly
    df['device_anomaly'] = df['device_id'].astype(str).str.contains("nan|N/A|NEW")

    return df

def add_anomaly_features(df):

    df['amount_anomaly'] = df['transaction_amount'] > df['transaction_amount'].quantile(0.99)

    df['invalid_ip'] = df['ip_address'].astype(str).str.contains("not_an_ip|nan|999")

    df['missing_payment'] = df['payment_method'].isna() | (df['payment_method'] == "")

    df['invalid_location'] = df['merchant_location'].astype(str).str.contains("#")

    df['location_mismatch'] = (
        df['user_location'].astype(str).str.lower() !=
        df['merchant_location'].astype(str).str.lower()
    )

    df['time_anomaly'] = df['is_night'] == 1
    
    df['pattern_flag'] = (
        (df['transaction_amount'] > 5000) &
        (df['location_mismatch'])
    )

    return df

def add_behavior_features(df):

    # 🔹 Transaction count per user
    df['user_txn_count'] = df.groupby('user_id')['transaction_id'].transform('count')

    # 🔹 Convert timestamp to datetime
    df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'], errors='coerce')

    # 🔹 Sort by user and time
    df = df.sort_values(by=['user_id', 'transaction_timestamp'])

    # 🔹 Time difference between transactions
    df['time_diff'] = df.groupby('user_id')['transaction_timestamp'].diff().dt.total_seconds()

    # 🔹 Fast transactions (less than 60 seconds)
    df['fast_txn'] = df['time_diff'] < 60

    # 🔹 Repeated rapid transactions per user
    df['rapid_txn_count'] = df.groupby('user_id')['fast_txn'].transform('sum')

    # 🔹 Flag suspicious repeated activity
    df['repeated_attempts'] = df['rapid_txn_count'] > 3

    return df