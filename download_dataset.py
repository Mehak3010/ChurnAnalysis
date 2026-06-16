
import pandas as pd

# Download the dataset directly
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Save locally
df.to_csv("Telco-Customer-Churn.csv", index=False)

print("Dataset downloaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
