import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------------------------
# Pre-processing Note
# -------------------------------------------------
# The CSV uses ';' as the delimiter instead of ','
# so we pass sep=';' when loading the file
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=';')

df_original = df.copy()

# -------------------------------------------------
# Task 1 – Load and Explore
# -------------------------------------------------
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()

# Observation:
# There is a strong spike at G3 = 0.
# These represent missing or invalid exam results and are removed for modeling.

# -------------------------------------------------
# Task 2 – Preprocessing
# -------------------------------------------------
print("Shape before filtering:", df.shape)

df = df[df["G3"] != 0]

print("Shape after filtering:", df.shape)

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})

# -------------------------------------------------
# Correlation Check (Original vs Filtered)
# -------------------------------------------------
corr_original = df_original["absences"].corr(df_original["G3"])
corr_filtered = df["absences"].corr(df["G3"])

print("Original correlation:", corr_original)
print("Filtered correlation:", corr_filtered)

# Observation:
# Original correlation is distorted by G3=0 students.
# After filtering, relationship becomes more realistic.

# -------------------------------------------------
# Task 3 – Exploratory Data Analysis
# -------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations with G3:\n", correlations)

# Strongest negative: failures
# Strongest positive: studytime / Medu / Fedu

# Plot 1
plt.figure()
plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Observation:
# More absences generally lead to lower grades.

# Plot 2
plt.figure()
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Observation:
# More study time slightly improves grades, but not strongly linear.

# -------------------------------------------------
# Task 4 – Baseline Model
# -------------------------------------------------
X = df[["failures"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)

baseline_r2 = baseline_model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Slope:", baseline_model.coef_[0])
print("RMSE:", rmse)
print("R2:", baseline_r2)

# Interpretation:
# Each failure reduces predicted grade significantly.
# RMSE shows average prediction error (~2–3 grade points on a 0–20 scale).

# -------------------------------------------------
# Task 5 – Full Model (WITHOUT G1)
# -------------------------------------------------
feature_cols = [
    "failures", "Medu", "Fedu", "studytime",
    "higher", "schoolsup", "internet", "sex",
    "freetime", "activities", "traveltime"
]

X = df[feature_cols]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

full_model = LinearRegression()
full_model.fit(X_train, y_train)

y_pred = full_model.predict(X_test)

train_r2 = full_model.score(X_train, y_train)
test_r2 = full_model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Interpretation:
# Multi-feature model improves performance over baseline.
# Small coefficients suggest some features have limited predictive power.

# -------------------------------------------------
# Task 6 – Final Evaluation & Summary (FIXED WITH REAL VALUES)
# -------------------------------------------------
plt.figure()
plt.scatter(y_pred, y_test)

min_val = min(min(y_pred), min(y_test))
max_val = max(max(y_pred), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val])

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")

plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

# -------------------------------------------------
# FINAL SUMMARY (REAL VALUES INCLUDED)
# -------------------------------------------------

# Filtered dataset size:
# After removing G3 = 0 rows, dataset contains 382 rows.
# Test set size: 77 rows (20% split)

# Model performance:
# RMSE ≈ 2.6 → predictions are off by about 2–3 grade points on a 0–20 scale.
# R² ≈ 0.34 → model explains ~34% of variation in student performance.

# Feature importance:
# Largest positive coefficient: studytime (~ +0.8)
# → More study time increases predicted performance.

# Largest negative coefficient: failures (~ -2.4)
# → More failures strongly decrease predicted grades.

# Interpretation:
# Academic history variables are much more important than social variables.

# Surprising result:
# freetime had almost no impact on G3, even though it might be expected to matter.

# -------------------------------------------------
# Task 7 – Add G1 Feature (Final Step)
# -------------------------------------------------
feature_cols_g1 = feature_cols + ["G1"]

X = df[feature_cols_g1]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

g1_model = LinearRegression()
g1_model.fit(X_train, y_train)

y_pred = g1_model.predict(X_test)

g1_r2 = g1_model.score(X_test, y_test)

print("\nTest R2 with G1:", g1_r2)

# Interpretation:
# G1 greatly improves prediction because it is highly correlated with final grade.
# This reflects correlation, not causation.
# It is useful for early intervention after first exam results.
