import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("uae_used_cars_10k.csv") 

features = ["Make", "Model", "Body Type", "Year", "Mileage"]
target = "Transmission"

df_cleaned = df.copy()
df_cleaned[["Make", "Model", "Body Type", "Transmission"]] = df_cleaned[
    ["Make", "Model", "Body Type", "Transmission"]
].fillna("Unknown")

df_cleaned[["Year", "Mileage"]] = df_cleaned[["Year", "Mileage"]].fillna(df_cleaned[["Year", "Mileage"]].median())
df_cleaned[["Year", "Mileage"]] = df_cleaned[["Year", "Mileage"]].astype(float)

label_encoders = {}
for col in ["Make", "Model", "Body Type"]:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    label_encoders[col] = le

transmission_encoder = LabelEncoder()
df_cleaned[target] = transmission_encoder.fit_transform(df_cleaned[target])  # Automatic = 0, Manual = 1

X = df_cleaned[features]
y = df_cleaned[target]

scaler = StandardScaler()
X.loc[:, ["Year", "Mileage"]] = scaler.fit_transform(X[["Year", "Mileage"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

df_misclassified = X_test.copy()
df_misclassified["Actual"] = y_test
df_misclassified["Predicted"] = y_pred
df_misclassified = df_misclassified[df_misclassified["Actual"] != df_misclassified["Predicted"]]

print("\nMisclassified Examples:")
print(df_misclassified.head())
