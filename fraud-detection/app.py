from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from collections import Counter
from utils.features import add_anomaly_features
from utils.features import add_behavior_features

from utils.cleaning import clean_data, data_quality_report
from utils.features import (
    create_features,
    generate_explanations,
    calculate_risk_score,
    detect_fraud_type
)

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

global_df = None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    df = pd.read_csv(file)

    # 🔹 Total rows from uploaded file
    total = len(df)

    # 🔹 Data processing
    df = clean_data(df)
    report = data_quality_report(df)
    df = create_features(df)
    df = add_behavior_features(df)
    df = add_anomaly_features(df)

    # 🔹 Features used for model
    features = [
        'transaction_amount',
        'amount_deviation',
        'is_new_device',
        'is_new_location',
        'is_night'
    ]

    # 🔹 Keep all rows
    df[features] = df[features].fillna(0)

    # 🔥 PROBABILITY-BASED PREDICTION
    probs_all = model.predict_proba(df[features])

    # Handle case when model has only one class
    if probs_all.shape[1] == 1:
        probs = probs_all[:, 0]   # fallback
    else:
        probs = probs_all[:, 1]   # normal case

    # 🔹 Risk score
    scores, levels = calculate_risk_score(df)
    df['risk_score'] = scores
    df['risk_level'] = levels

    # 🔹 Convert probability to prediction
    df['predicted'] = (probs > 0.5).astype(int)

    df['anomaly_count'] = (
        df['amount_anomaly'].astype(int) +
        df['invalid_ip'].astype(int) +
        df['invalid_location'].astype(int) +
        df['missing_payment'].astype(int) +
        df['location_mismatch'].astype(int) +
        df['time_anomaly'].astype(int) +
        (df['user_txn_count'] > df['user_txn_count'].quantile(0.95)).astype(int) +
        df['fast_txn'].astype(int) +
        df['repeated_attempts'].astype(int)
    )

    # 🔹 FINAL FRAUD SCORE
    df['final_score'] = (
        df['risk_score'] * 0.5 +
        (probs * 100) * 0.3 +
        (df['anomaly_count'] * 12) * 0.15 +
        (df['pattern_flag'].astype(int) * 20)
    )
    
    
    # 🔹 Adaptive percentage based on dataset size
    if len(df) < 5000:
        top_percent = 0.10
    elif len(df) < 50000:
        top_percent = 0.05
    else:
        top_percent = 0.02

    top_n = int(len(df) * top_percent)

    # 🔹 Select top suspicious transactions
    df['fraud'] = 0
    df.loc[df['final_score'].nlargest(top_n).index, 'fraud'] = 1

    # 🔹 Confidence
    df['confidence'] = probs * 100

    # 🔹 Explanation
    df['explanation'] = generate_explanations(df)

    # 🔹 Fraud type
    df['fraud_type'] = detect_fraud_type(df)
    fraud_type_counts = Counter(df['fraud_type'])

    # 🔹 Charts folder
    if not os.path.exists("static/charts"):
        os.makedirs("static/charts")

    # -------------------------------
    # Chart 1: Fraud vs Safe
    # -------------------------------
    fraud_counts = df['fraud'].value_counts()

    plt.figure()
    fraud_counts.plot(kind='bar')
    plt.title("Fraud vs Safe Transactions")
    plt.xlabel("Class (0 = Safe, 1 = Fraud)")
    plt.ylabel("Count")
    plt.savefig("static/charts/fraud_bar.png")
    plt.close()

    # -------------------------------
    # Chart 2: Fraud by Hour
    # -------------------------------
    fraud_by_hour = df.groupby('hour')['fraud'].sum()

    plt.figure()
    fraud_by_hour.plot(kind='line')
    plt.title("Fraud Transactions by Hour")
    plt.xlabel("Hour")
    plt.ylabel("Fraud Count")
    plt.savefig("static/charts/fraud_hour.png")
    plt.close()

    # -------------------------------
    # Chart 3: Pie Chart
    # -------------------------------
    plt.figure()
    df['fraud'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Fraud Distribution")
    plt.ylabel("")
    plt.savefig("static/charts/fraud_pie.png")
    plt.close()

    # 🔹 Final counts
    fraud_count = int(df['fraud'].sum())
    safe = total - fraud_count
    fraud_rate = round((fraud_count / total) * 100, 2)

    # 🔹 Top reasons
    all_reasons = []
    for exp in df['explanation']:
        parts = exp.split(", ")
        all_reasons.extend(parts)

    reason_counts = Counter(all_reasons)
    top_reasons = dict(reason_counts.most_common(3))

    # 🔹 Store for download
    global global_df
    global_df = df

    # 🔹 Return to UI
    return render_template(
        'result.html',
        total=total,
        fraud=fraud_count,
        safe=safe,
        rate=fraud_rate,
        data=df.head(20).to_dict(orient='records'),
        report=report,
        top_reasons=top_reasons,
        fraud_types=fraud_type_counts
    )


@app.route('/download')
def download():
    global global_df

    file_path = "static/report.csv"
    global_df.to_csv(file_path, index=False)

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # Use the port assigned by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
