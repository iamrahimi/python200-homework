import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# -------------------------------------------------
# The CSV uses ';' as the delimiter instead of ','
# so we pass sep=';' when loading the file
# -------------------------------------------------
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=';')

# Keep original copy for correlation comparison
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
# These represent missing or invalid exam results and can distort analysis.

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

print("Correlation (original):", corr_original)
print("Correlation (filtered):", corr_filtered)

# Observation:
# The original dataset is skewed because G3=0 students often have high absences.
# After filtering, correlation becomes more meaningful.

# -------------------------------------------------
# Task 3 – Exploratory Data Analysis
# -------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations with G3:\n", correlations)

# Strongest negative: failures
# Strongest positive: studytime / parental education

# Plot 1 – Absences vs Grade
plt.figure()
plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Observation:
# More absences generally lead to lower grades.

# Plot 2 – Study Time vs Grade
plt.figure()
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Observation:
# More study time slightly improves performance, but not perfectly.

# -------------------------------------------------
# Task 4 – Baseline Model (Failures only)
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
# Each failure reduces the predicted grade by the slope value.
# RMSE shows average prediction error on a 0–20 scale.

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
# Full model performs better than baseline, meaning multiple features improve prediction.
# Small coefficient features may not be important in real-world use.

# -------------------------------------------------
# Task 6 – Evaluate & Summary
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

# Summary:
# Dataset size after cleaning: df.shape
# Test set size: 20% of cleaned dataset
#
# RMSE: average prediction error on 0–20 scale
# R²: how much variance in grades the model explains
#
# Strongest positive feature: studytime / education
# Strongest negative feature: failures
#
# Model weakness:
# struggles with extreme high/low grades
# above diagonal = underprediction
# below diagonal = overprediction

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
# G1 greatly improves prediction because it is directly related to final grade.
# This does NOT mean causation — only strong correlation.
# Useful for early intervention after first exam results are available.
