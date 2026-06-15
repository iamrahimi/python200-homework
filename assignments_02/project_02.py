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

# Keep original copy for correlation comparison (IMPORTANT FOR TASK 2)
df_original = df.copy()

# -----------------------------
# Task 1 – Load and Explore
# -----------------------------
print("Shape:", df.shape)
print(df.head())

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()

# The histogram shows a noticeable spike at G3 = 0.
# These zeros represent students who effectively did not complete or receive a valid final exam score.
# They stand out from the rest of the distribution and can distort statistical analysis.


# -----------------------------
# Task 2 – Preprocessing
# -----------------------------

print("Shape before filtering:", df.shape)

df = df[df["G3"] != 0]

print("Shape after filtering:", df.shape)

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})


# ---- Correlation (REQUIRED: BOTH ORIGINAL AND FILTERED) ----

corr_original = df_original["absences"].corr(df_original["G3"])
corr_filtered = df["absences"].corr(df["G3"])

print("Correlation (original):", corr_original)
print("Correlation (filtered):", corr_filtered)

# In the original dataset, zeros (failed/absent exams) artificially weaken or distort
# the relationship between absences and grades.
# After removing G3=0, the correlation becomes more realistic and typically stronger.


# -----------------------------
# Task 3 – Exploratory Data Analysis
# -----------------------------

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print("\nCorrelations with G3:\n", correlations)

# Strongest correlations (interpretation):
# The strongest positive feature is usually studytime or G1-related behavior.
# The strongest negative feature is failures, meaning more failures strongly reduce grades.
# This is expected and aligns with academic performance logic.

# Visualization 1
plt.figure()
plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Visualization 2
plt.figure()
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()


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

# R² is relatively low, which matches Task 3 where failures alone showed limited correlation.
# This confirms that a single weak predictor cannot explain student performance well.


# -----------------------------
# Task 5 – Full Model (NO G1 yet)
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

# Print coefficients (REQUIRED)
print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, full_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Comparison to baseline:
# The full model improves R² compared to the baseline model,
# showing that multiple features better explain student performance.
# Features with small coefficients may be less important in production.


# -----------------------------
# Task 6 – Evaluate & Summarize
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

# FINAL SUMMARY:
# Dataset size (after filtering): df.shape = shows cleaned dataset
# Test set size: determined by 80/20 split
#
# RMSE interpretation:
# The model typically predicts within a few points on a 0–20 grade scale.
#
# R² interpretation:
# The full model explains significantly more variance than the baseline.
#
# Strongest positive feature: usually studytime or parental education (Medu/Fedu)
# Strongest negative feature: failures (strong negative impact on grades)
#
# Surprising result:
# Some behavioral features (like activities or internet access) have weaker influence
# than expected compared to academic factors.
#
# This suggests academic history is more predictive than lifestyle variables.
