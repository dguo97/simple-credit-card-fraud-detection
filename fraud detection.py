import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

# --- STEP 1: LOAD OR CREATE A CLEAN FRAUD DATASET ---
# If using a downloaded Kaggle file, replace "fraud_data.csv" with your file path.
# For demonstration purposes, we simulate a clean operational dataset with human-readable attributes.
print("Loading transaction dataset...")

try:
  df = pd.read_csv("fraud_data.csv")
except FileNotFoundError:
  print("Sample file not found. Generating a simulated clean transactional dataset...")
  np.random.seed(42)
  n_samples = 5000
  df = pd.DataFrame(
      {
          "Transaction_Amount": np.random.exponential(scale=60, size=n_samples),
          "Merchant_Category": np.random.choice(
              ["Grocery", "Electronics", "Travel", "Dining", "Retail"],
              size=n_samples,
              p=[0.4, 0.15, 0.05, 0.2, 0.2],
          ),
          "Transaction_Type": np.random.choice(
              ["In-Store", "Online", "ATM"], size=n_samples, p=[0.5, 0.4, 0.1]
          ),
          "Is_Weekend": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
      }
  )
  # Inject realistic fraud rules for labels
  fraud_cond = (df["Transaction_Amount"] > 800) | (
      (df["Merchant_Category"] == "Travel")
      & (df["Transaction_Type"] == "Online")
      & (df["Transaction_Amount"] > 300)
  )
  df["Fraud_Label"] = np.where(
      fraud_cond & (np.random.rand(n_samples) > 0.3), 1, 0
  )

print(f"Dataset loaded! Total rows: {len(df)}")

# --- STEP 2: PREPROCESSING & ENCODING ---
print("Encoding categorical fields and scaling features...")

# Initialize encoders and scalers
le_cat = LabelEncoder()
le_type = LabelEncoder()
scaler = RobustScaler()

df["Merchant_Category_Enc"] = le_cat.fit_transform(df["Merchant_Category"])
df["Transaction_Type_Enc"] = le_type.fit_transform(df["Transaction_Type"])
df["Amount_Scaled"] = scaler.fit_transform(df[["Transaction_Amount"]])

feature_cols = [
    "Amount_Scaled",
    "Merchant_Category_Enc",
    "Transaction_Type_Enc",
    "Is_Weekend",
]
X = df[feature_cols]
y = df["Fraud_Label"]

# --- STEP 3: TRAIN ISOLATION FOREST & HYBRID CLASSIFIER ---
print("Training Isolation Forest and Hybrid Random Forest models...")
iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
iso.fit(X)

# Extract raw anomaly scores as an additional feature
df["anomaly_score"] = iso.decision_function(X)
X_hybrid = df[feature_cols + ["anomaly_score"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_train, y_train)
print("Model training complete!\n")

# --- STEP 4: INTERACTIVE USER INPUT PREDICTION FUNCTION ---


def check_user_transaction(amount, category, trans_type, is_weekend):
  """Takes intuitive human inputs, processes them, and predicts fraud status."""
  # Convert user input into DataFrame matching training format
  input_df = pd.DataFrame(
      {
          "Transaction_Amount": [amount],
          "Merchant_Category": [str(category)],
          "Transaction_Type": [str(trans_type)],
          "Is_Weekend": [int(is_weekend)],
      }
  )

  # Transform features
  input_df["Merchant_Category_Enc"] = le_cat.transform(
      input_df["Merchant_Category"]
  )
  input_df["Transaction_Type_Enc"] = le_type.transform(
      input_df["Transaction_Type"]
  )
  input_df["Amount_Scaled"] = scaler.transform(input_df[["Transaction_Amount"]])

  features_only = input_df[
      [
          "Amount_Scaled",
          "Merchant_Category_Enc",
          "Transaction_Type_Enc",
          "Is_Weekend",
      ]
  ]

  # Get Isolation Forest anomaly score
  user_anomaly_score = iso.decision_function(features_only)

  # Construct hybrid feature set
  input_hybrid = features_only.copy()
  input_hybrid["anomaly_score"] = user_anomaly_score

  # Predict via Random Forest
  prediction = rf.predict(input_hybrid)[0]
  probability = rf.predict_proba(input_hybrid)[0][1]

  # Output Results
  print("=" * 45)
  print("🔍 TRANSACTION FRAUD EVALUATION")
  print("=" * 45)
  print(f"Amount: ${amount:.2f} | Category: {category}")
  print(f"Type: {trans_type} | Weekend: {bool(is_weekend)}")
  print(f"Fraud Probability: {probability:.2%}")

  if prediction == 1 or probability > 0.5:
    print("Verdict: 🚨 High Risk / Potential Fraud Detected!")
  else:
    print("Verdict: ✅ Normal Transaction")
  print("=" * 45 + "\n")


# --- STEP 5: TEST USER INPUTS WITH READABLE WORDS ---
print("Testing interactive user transactions...")

# Test 1: Normal everyday grocery purchase
check_user_transaction(
    amount=35.50, category="Grocery", trans_type="In-Store", is_weekend=0
)

# Test 2: Suspicious large online travel transaction
check_user_transaction(
    amount=1450.00, category="Travel", trans_type="Online", is_weekend=1
)