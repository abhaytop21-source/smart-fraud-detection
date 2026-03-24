from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

def train_model(df):

    # 🔹 Create synthetic fraud labels (VERY IMPORTANT)
    df['fraud'] = np.where(
        (
            (df['amount_deviation'] > 4) & 
            (df['is_new_device'] == 1)
        ) |
        (
            (df['is_new_location'] == 1) & 
            (df['is_night'] == 1)
        ),
        1, 0
    )

    # 🔹 Features
    features = [
        'transaction_amount',
        'amount_deviation',
        'is_new_device',
        'is_new_location',
        'is_night'
    ]

    X = df[features]
    y = df['fraud']

    # 🔹 Handle missing values
    X = X.fillna(0)

    # 🔥 Improved model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    # 🔹 Train
    model.fit(X, y)

    # 🔹 Save model
    joblib.dump(model, 'model/model.pkl')

    print("✅ Model trained successfully!")