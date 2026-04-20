import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# The dataset uses semicolons (;) as separators instead of commas,
# so we must specify sep=';' when loading the CSV file with pandas.
df = pd.read_csv("assignments_02/student_performance_math.csv",sep=';')
df = df.drop(columns=["G1", "G2"])

# print(df.head())

# Task 1: Load and Explore
# Print shape
print("Shape:", df.shape)

# Print first five rows
print(df.head())

# Print data types
print(df.dtypes)

# Histogram of G3
plt.figure()

plt.hist(df["G3"], bins=21)

plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()


# Task 2: Preprocess the Data

# 2. Remove invalid rows (G3 = 0)
df = df[df["G3"] != 0]

# 3. Convert categorical columns

# Yes/No columns → 1/0
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# Sex → 0/1
df["sex"] = df["sex"].map({"F": 0, "M": 1})

# 4. Define features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R2:", r2)



# Task 3: Exploratory Data Analysis

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})


# Correlation with G3

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlations with G3
correlations = numeric_df.corr()["G3"].sort_values()

print("Correlations with G3 (sorted):\n")
print(correlations)


#Q4 

# Feature and target
X = df[["failures"]]   # must be 2D
y = df["G3"]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Metrics
slope = model.coef_[0]
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("Slope:", slope)
print("RMSE:", rmse)
print("R2:", r2)


#Q5
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

df_clean = df[df["G3"] != 0].copy()

# Convert yes/no safely BEFORE anything else
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df_clean[col] = (
        df_clean[col]
        .astype(str)          # ensure string
        .str.strip()          # remove spaces
        .str.lower()          # make lowercase
        .map({"yes": 1, "no": 0})
    )

# Convert sex
df_clean["sex"] = (
    df_clean["sex"]
    .astype(str)
    .str.strip()
    .str.upper()
    .map({"F": 0, "M": 1})
)


# Features and target
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
                        "internet", "sex", "freetime", "activities", "traveltime", "G1"]

df_clean = df_clean.dropna(subset=["G3"] + feature_cols)

X = df_clean[feature_cols].values
y = df_clean["G3"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

# Coefficients
print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")


# Task 6: Evaluate and Summarize

# Predictions (from your full model)
y_pred = model.predict(X_test)

# Plot
plt.figure()

plt.scatter(y_pred, y_test)

# Diagonal line (perfect prediction)
min_val = min(min(y_pred), min(y_test))
max_val = max(max(y_pred), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()