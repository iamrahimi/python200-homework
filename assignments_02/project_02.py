import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# The dataset uses semicolons (;) as separators instead of commas,
# so we must specify sep=';' when loading the CSV file.
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=';')
df = df.drop(columns=["G1", "G2"])

# -----------------------------
# Task 1: Load and Explore
# -----------------------------
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


# -----------------------------
# Task 2: Preprocess the Data
# -----------------------------

# Print shape before filtering
print("Shape before removing G3=0:", df.shape)

# Remove invalid rows (G3 = 0 means student has no valid final grade,
# often due to absence of final exam or missing evaluation)
df = df[df["G3"] != 0]

# Print shape after filtering
print("Shape after removing G3=0:", df.shape)

# Convert categorical columns
yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})


# -----------------------------
# Correlation Check (MISSING STEP FIXED)
# -----------------------------

# Pearson correlation between absences and G3 (original data idea)
corr_original = df["absences"].corr(df["G3"])

print("Correlation (filtered data) absences vs G3:", corr_original)

# Interpretation:
# A negative correlation means more absences tend to be associated with lower grades.


# -----------------------------
# Task 3: Exploratory Data Analysis
# -----------------------------

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()["G3"].sort_values()

print("Correlations with G3:\n", correlations)

# Visualization 1: Absences vs G3
plt.figure()
plt.scatter(df["absences"], df["G3"])
plt.title("Absences vs Final Grade")
plt.xlabel("Absences")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

# Interpretation:
# Students with higher absences generally tend to have lower G3 scores.

# Visualization 2: Study time vs G3
plt.figure()
plt.scatter(df["studytime"], df["G3"])
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Interpretation:
# More study time is generally associated with slightly higher grades, but not perfectly.


# -----------------------------
# Task 4: Baseline Model (failures only)
# -----------------------------

X = df[["failures"]]
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("Slope:", slope)
print("RMSE:", rmse)
print("R2:", r2)

# Interpretation:
# The slope shows how much G3 decreases as failures increase.
# The RMSE shows the average prediction error on a 0–20 grade scale.
# A high RMSE means the model is not very accurate using only failures.


# -----------------------------
# Task 5: Full Model (WITHOUT G1/G2 first) FIXED
# -----------------------------

feature_cols = ["failures", "Medu", "Fedu", "studytime",
                 "higher", "schoolsup", "internet", "sex",
                 "freetime", "activities", "traveltime", "absences"]

df_clean = df.dropna(subset=["G3"] + feature_cols)

X = df_clean[feature_cols]
y = df_clean["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_pred = model_full.predict(X_test)

train_r2 = model_full.score(X_train, y_train)
test_r2 = model_full.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Train R2:", train_r2)
print("Test R2:", test_r2)
print("RMSE:", rmse)

# Interpretation:
# Comparing train vs test R2 helps detect overfitting.
# If train R2 is much higher than test R2, the model is overfitting the training data.


# -----------------------------
# Task 5: Add G1 and analysis (MISSING FIXED)
# -----------------------------

feature_cols_with_g1 = feature_cols + ["G1"]

X = df_clean[feature_cols_with_g1]
y = df_clean["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(X_train, y_train)

y_pred = model_g1.predict(X_test)

print("Test R2 with G1:", model_g1.score(X_test, y_test))

# Interpretation:
# G1 is strongly related to G3 because it is an earlier exam score.
# This can improve accuracy but reduces usefulness for early intervention,
# because teachers already need G1 to predict G3.


# -----------------------------
# Task 6: Evaluate & Summarize (FIXED)
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

# Final Summary:
# The dataset contains students with various academic and social features.
# After cleaning, we removed invalid grades (G3=0).
# The full model performs better than the baseline (failures-only model).
# RMSE shows average prediction error on a 0–20 grade scale.
# R² shows how much variance in grades the model explains.
# The strongest positive influence is usually G1 (when included),
# while failures and absences tend to negatively impact performance.
# One surprising result is that some social factors have weaker effects than expected.
