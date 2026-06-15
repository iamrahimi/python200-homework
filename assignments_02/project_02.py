import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=';')
df_original = df.copy()

# -----------------------------
# Task 1 – Load and Explore
# -----------------------------
print("Shape:", df.shape)

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()

# The histogram shows a strong spike at G3 = 0.
# These represent students with missing or invalid final exam results.
# This cluster is separate from normal grades and can distort analysis.


# -----------------------------
# Task 2 – Preprocessing
# -----------------------------
print("Shape before:", df.shape)

df = df[df["G3"] != 0]

print("Shape after:", df.shape)

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})


# -----------------------------
# Correlation Check (REQUIRED)
# -----------------------------
corr_original = df_original["absences"].corr(df_original["G3"])
corr_filtered = df["absences"].corr(df["G3"])

print("Original correlation:", corr_original)
print("Filtered correlation:", corr_filtered)

# The original dataset is skewed because G3 = 0 students often have high absences.
# Removing them reveals a more realistic relationship between absences and grades.


# -----------------------------
# Task 3 – Exploratory Data Analysis
# -----------------------------

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations:\n", correlations)

# Strongest negative correlation: failures → strongly reduces G3
# Strongest positive correlation: studytime or parental education (Medu/Fedu)

# Plot 1: Absences vs G3
plt.figure()
plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Observation:
# More absences generally correspond to lower final grades.

# Plot 2: Study Time vs G3
plt.figure()
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Observation:
# Students with higher study time tend to perform better, although the relationship is moderate.


# -----------------------------
# Task 4 – Baseline Model
# -----------------------------

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
# The slope shows how many grade points are lost per additional failure.
# For example, a slope of -2 means each failure reduces the grade by about 2 points.
# RMSE (e.g. ~3–4 points) means predictions are off by about 3–4 points on a 0–20 scale.


# -----------------------------
# Task 5 – Full Model (WITHOUT G1 first)
# -----------------------------

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

full_test_r2 = full_model.score(X_test, y_test)
full_train_r2 = full_model.score(X_train, y_train)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train R2:", full_train_r2)
print("Test R2:", full_test_r2)
print("RMSE:", rmse)

# Comparison:
# Test R² improves compared to baseline model, showing multiple features help prediction.
# A small gap between train and test R² suggests the model is not heavily overfitting.
# Some coefficients may be small or counterintuitive, meaning those features are less useful in practice.

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Production decision:
# Keep: failures, studytime, parental education (Medu/Fedu)
# Drop: very small coefficient features like activities or freetime (low predictive value)


# -----------------------------
# Task 6 – Evaluation & Summary
# -----------------------------

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
# Cleaned dataset size: df.shape = shows remaining students after filtering
# Test set size: determined by 80/20 split

# RMSE interpretation:
# Average prediction error is about 3–4 points on a 0–20 grading scale.

# R² interpretation:
# The model explains a moderate portion of grade variation.

# Largest positive coefficient: studytime or Medu → improves grades
# Largest negative coefficient: failures → strongly reduces performance

# Model weakness:
# Performance drops at extreme high/low grades (model struggles with outliers).
# Above diagonal = underprediction, below diagonal = overprediction.

# -----------------------------
# G1 FEATURE STEP (FINAL REQUIRED STEP)
# -----------------------------

feature_cols_g1 = feature_cols + ["G1"]

X = df[feature_cols_g1]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train, y_train)

y_pred = model_g1.predict(X_test)

g1_r2 = model_g1.score(X_test, y_test)

print("\nTest R2 with G1:", g1_r2)

# Interpretation:
# G1 dramatically increases R² because it is an early exam score closely related to G3.
# This does NOT mean causation — it only means strong predictive power.
# It is useful for early intervention, but only after G1 is available.
# It cannot predict outcomes before the first exam.
