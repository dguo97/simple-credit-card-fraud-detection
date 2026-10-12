import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler

# --- STEP 1: SETUP DATA & TRAIN MODELS ---
print("Initializing fraud detection model...")
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

fraud_cond = (df["Transaction_Amount"] > 800) | (
    (df["Merchant_Category"] == "Travel")
    & (df["Transaction_Type"] == "Online")
    & (df["Transaction_Amount"] > 300)
)
df["Fraud_Label"] = np.where(fraud_cond & (np.random.rand(n_samples) > 0.3), 1, 0)

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

iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
iso.fit(X)

df["anomaly_score"] = iso.decision_function(X)
X_hybrid = df[feature_cols + ["anomaly_score"]]

rf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_hybrid, y)
print("Model ready!\n")

# --- STEP 2: INTERACTIVE GUIDED INPUT FUNCTION ---


def prompt_user_transaction():
  """Guides the user step-by-step with clear instructions and validation."""
  print("=" * 50)
  print("💳 CREDIT CARD FRAUD DETECTION - INTERACTIVE CHECKER")
  print("=" * 50)
  print(
      "Please enter the transaction details below when prompted.\n"
  )

  # 1. Transaction Amount
  while True:
    try:
      amount = float(
          input(
              "1. Enter Transaction Amount (e.g., 45.50 or 1200.00): $"
          ).strip()
      )
      if amount < 0:
        print("Amount cannot be negative. Try again.")
        continue
      break
    except ValueError:
      print("⚠️ Invalid input. Please enter a valid number.")

  # 2. Merchant Category
  valid_categories = ["Grocery", "Electronics", "Travel", "Dining", "Retail"]
  print(f"\n2. Available Merchant Categories: {', '.join(valid_categories)}")
  while True:
    category = input("Select or type a category from the list above: ").strip()
    # Capitalize first letter to match nicely
    category = category.capitalize()
    if category in valid_categories:
      break
    print(
        f"⚠️ Invalid choice. Please choose strictly from: {', '.join(valid_categories)}"
    )

  # 3. Transaction Type
  valid_types = ["In-Store", "Online", "ATM"]
  print(f"\n3. Available Transaction Types: {', '.join(valid_types)}")
  while True:
    trans_type = input("Select transaction type from the list above: ").strip()
    # Format case to match choices
    if trans_type.lower() == "in-store":
      trans_type = "In-Store"
    else:
      trans_type = trans_type.capitalize()

    if trans_type in valid_types:
      break
    print(f"⚠️ Invalid choice. Please choose strictly from: {', '.join(valid_types)}")

  # 4. Weekend Flag
  while True:
    weekend_input = (
        input("\n4. Is this transaction taking place on a weekend? (y/n): ")
        .strip()
        .lower()
    )
    if weekend_input in ["y", "yes"]:
      is_weekend = 1
      break
    elif weekend_input in ["n", "no"]:
      is_weekend = 0
      break
    print("⚠️ Please type 'y' for yes or 'n' for no.")

  # --- PROCESS AND PREDICT ---
  input_df = pd.DataFrame(
      {
          "Transaction_Amount": [amount],
          "Merchant_Category": [category],
          "Transaction_Type": [trans_type],
          "Is_Weekend": [is_weekend],
      }
  )

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
  user_anomaly_score = iso.decision_function(features_only)

  input_hybrid = features_only.copy()
  input_hybrid["anomaly_score"] = user_anomaly_score

  prediction = rf.predict(input_hybrid)[0]
  probability = rf.predict_proba(input_hybrid)[0][1]

  # --- DISPLAY RESULTS ---
  print("\n" + "=" * 50)
  print("📊 ANALYSIS RESULTS")
  print("=" * 50)
  print(f"• Amount: ${amount:.2f}")
  print(f"• Category: {category}")
  print(f"• Type: {trans_type}")
  print(f"• Weekend: {'Yes' if is_weekend == 1 else 'No'}")
  print(f"• Calculated Fraud Probability: {probability:.2%}")
  print("-" * 50)

  if prediction == 1 or probability > 0.5:
    print("🚨 VERDICT: High Risk / Potential Fraud Detected!")
  else:
    print("✅ VERDICT: Normal Transaction")
  print("=" * 50 + "\n")


# Run the interactive prompt function
if __name__ == "__main__":
  prompt_user_transaction()