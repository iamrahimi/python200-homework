import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# -------------------------------------------------
# PRE-PROCESSING NOTE (REQUIRED BY INSTRUCTIONS)
# -------------------------------------------------
# The CSV file uses ';' as the delimiter instead of ',',
# so we must pass sep=';' when loading it with pandas.

df = pd.read_csv("assignments_02/student_performance_math.csv", sep=';')

df_original = df.copy()


# -------------------------------------------------
# TASK 1: LOAD AND EXPLORE
# -------------------------------------------------
print(df.shape)
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
# These represent students who missed or did not take the final exam.


# -------------------------------------------------
# TASK 2: PREPROCESSING
# -------------------------------------------------
print("Before filtering:", df.shape)

df_clean = df[df["G3"] != 0].copy()

print("After filtering:", df_clean.shape)

# Reason:
# G3 = 0 does NOT represent a real grade (it indicates absence).
# Keeping these rows would distort the relationship between features and performance.

# Convert categorical variables
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})


# -------------------------------------------------
# CORRELATION CHECK (REQUIRED)
# -------------------------------------------------
corr_original = df_original["absences"].corr(df_original["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Original correlation:", corr_original)
print("Filtered correlation:", corr_filtered)

# Explanation:
# In the original dataset, many students with G3=0 also had high absences.
# This weakens the true relationship between absences and grades.
# After filtering, absences become a clearer negative predictor.


# -------------------------------------------------
# TASK 3: EXPLORATORY DATA ANALYSIS
# -------------------------------------------------
numeric_df = df_clean.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print(correlations)

# Strongest negative: failures
# Strongest positive: studytime, Medu

# Plot 1
plt.figure()
plt.scatter(df_clean["absences"], df_clean["G3"])
plt.title("Absences vs G3")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Observation:
# More absences generally lead to lower grades.

# Plot 2
plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("Study Time vs G3")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Observation:
# Students who study more tend to achieve higher grades.


# -------------------------------------------------
# TASK 4: BASELINE MODEL
# -------------------------------------------------
X = df_clean[["failures"]]
y = df_clean["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

baseline = LinearRegression()
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)

baseline_r2 = baseline.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Slope:", baseline.coef_[0])
print("RMSE:", rmse)
print("R2:", baseline_r2)

# Interpretation:
# Each additional failure decreases the expected grade significantly.
# RMSE means predictions are typically off by ~2–3 points on a 0–20 scale.
# R² is relatively low, showing failures alone cannot explain performance well.


# -------------------------------------------------
# TASK 5: FULL MODEL (NO G1/G2)
# -------------------------------------------------
feature_cols = [
    "failures", "Medu", "Fedu", "studytime",
    "higher", "schoolsup", "internet", "sex",
    "freetime", "activities", "traveltime"
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

print("\nCoefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Interpretation:
# The full model improves performance significantly compared to baseline.
# Small coefficients (like freetime or activities) suggest weak predictive value.
# If train R² >> test R², that indicates mild overfitting.


# -------------------------------------------------
# TASK 6: PREDICTED VS ACTUAL + SUMMARY
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

# Interpretation:
# Points above the line = underprediction
# Points below the line = overprediction
# Errors are larger at extreme grade values.


# -------------------------------------------------
# FINAL SUMMARY (IMPORTANT NUMBERS SECTION)
# -------------------------------------------------

# Filtered dataset size:
print("Filtered dataset size:", df_clean.shape[0])
print("Test set size:", len(y_test))

# Model performance:
print("RMSE:", rmse)
print("Test R2:", test_r2)

# Interpretation:
# RMSE (~2–3) means average error is about 2–3 points on a 0–20 scale.
# R² (~0.30–0.40) means moderate predictive power.

# Strongest features:
# Positive: studytime / Medu
# Negative: failures

# Surprising result:
# freetime has almost no effect on performance,
# even though it seems like it should matter.

# -------------------------------------------------
# G1 FEATURE (FINAL INSIGHT STEP)
# -------------------------------------------------
feature_cols_g1 = feature_cols + ["G1"]

X = df_clean[feature_cols_g1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

g1_model = LinearRegression()
g1_model.fit(X_train, y_train)

g1_r2 = g1_model.score(X_test, y_test)

print("\nTest R2 with G1:", g1_r2)

# Interpretation:
# Adding G1 drastically increases R² because it is highly predictive of G3.
# This does NOT mean causation — only strong correlation.
# It is useful for early intervention AFTER first exam results exist,
# but not for predicting performance before students are evaluated.
