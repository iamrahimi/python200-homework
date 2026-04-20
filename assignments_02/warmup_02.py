import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

# --- scikit-learn API ---
# Q1


# Data
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Step 1: Create model
model = LinearRegression()

# Step 2: Fit model
model.fit(years, salary)

# Step 3: Predict
pred_4_years = model.predict([[4]])
pred_8_years = model.predict([[8]])

# Outputs
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", pred_4_years[0])
print("Predicted salary for 8 years:", pred_8_years[0])


# Q2

# 1D array (single feature)
x = np.array([10, 20, 30, 40, 50])

# Print original shape
print("Original shape:", x.shape)

# Convert to 2D array (required for scikit-learn)
x_2d = x.reshape(-1, 1)

# Print new shape
print("New shape:", x_2d.shape)


# Q3

# Generate synthetic dataset
X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

# Create KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X_clusters)

# Predict cluster labels
labels = kmeans.labels_

# Print cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Count points in each cluster
print("Points per cluster:", np.bincount(labels))

# Plot the clusters
plt.figure()

plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="black",
    marker="X",
    s=200,
    label="Centers"
)

plt.title("K-Means Clustering (3 Clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Save figure
plt.savefig("assignments_02/outputs/kmeans_clusters.png")

plt.show()

# --- Linear Regression API ---

np.random.seed(42)

num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Q1

# Make sure output folder exists
os.makedirs("outputs", exist_ok=True)

# Scatter plot: Age vs Cost, colored by smoker status
plt.figure()

plt.scatter(age, cost, c=smoker, cmap="coolwarm")

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")

plt.savefig("assignments_02/outputs/cost_vs_age.png")
plt.show()

# Q2

# Use age as the only feature (must be 2D for scikit-learn)
X = age.reshape(-1, 1)
y = cost

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Slope and intercept
slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

# Predictions on test set
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

# R2 score
r2 = model.score(X_test, y_test)
print("R2:", r2)

# Q4

# Combine features (age + smoker)
X_full = np.column_stack([age, smoker])
y = cost

# Split data (same 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Fit model
model_full = LinearRegression()
model_full.fit(X_train, y_train)

# Predictions
y_pred_full = model_full.predict(X_test)

# R2 score
r2_full = model_full.score(X_test, y_test)
print("R2 with age + smoker:", r2_full)

# Coefficients
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Q5

# Predictions from the full model (age + smoker)
y_pred_full = model_full.predict(X_test)

# Create plot
plt.figure()

# Scatter: predicted vs actual
plt.scatter(y_pred_full, y_test)

# Diagonal reference line (perfect predictions)
min_val = min(min(y_pred_full), min(y_test))
max_val = max(max(y_pred_full), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], color="red")

# Labels and title
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

# Save figure
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()