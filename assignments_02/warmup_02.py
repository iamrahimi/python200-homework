# --- scikit-learn API ---

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


# -------------------------------------------------
# scikit-learn Question 1
# -------------------------------------------------

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4_years = model.predict([[4]])
pred_8_years = model.predict([[8]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Prediction (4 years):", pred_4_years[0])
print("Prediction (8 years):", pred_8_years[0])


# -------------------------------------------------
# scikit-learn Question 2
# -------------------------------------------------

x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

# scikit-learn requires X to be 2D because it expects data in the form
# (number of samples, number of features), even if there is only one feature.
print("New shape:", x_2d.shape)


# -------------------------------------------------
# scikit-learn Question 3 (K-Means)
# -------------------------------------------------

X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)

labels = kmeans.labels_

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

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

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/kmeans_clusters.png")
plt.show()


# -------------------------------------------------
# Linear Regression Dataset (shared)
# -------------------------------------------------

np.random.seed(42)

num_patients = 100

age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# -------------------------------------------------
# Linear Regression Question 1
# -------------------------------------------------

os.makedirs("outputs", exist_ok=True)

plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")

plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")

plt.savefig("outputs/cost_vs_age.png")
plt.show()

# Observation:
# Two visible groups appear: smokers and non-smokers.
# Smokers consistently have higher medical costs, showing smoker is a strong predictor.


# -------------------------------------------------
# Linear Regression Question 2
# -------------------------------------------------

X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# -------------------------------------------------
# Linear Regression Question 3
# -------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]
intercept = model.intercept_

print("Slope:", slope)
print("Intercept:", intercept)

y_pred = model.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model.score(X_test, y_test)

print("RMSE:", rmse)
print("R2:", r2)

# Interpretation:
# The slope shows how much cost increases per additional year of age.
# For example, a slope of ~200 means each year increases cost by about $200.


# -------------------------------------------------
# Linear Regression Question 4
# -------------------------------------------------

X_full = np.column_stack([age, smoker])

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_pred_full = model_full.predict(X_test)

r2_full = model_full.score(X_test, y_test)

print("R2 with age + smoker:", r2_full)

print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# Interpretation:
# The smoker coefficient represents how much more a smoker pays compared to a non-smoker,
# holding age constant. It is a large positive value, showing smoking increases cost significantly.


# -------------------------------------------------
# Linear Regression Question 5
# -------------------------------------------------

plt.figure()

plt.scatter(y_pred_full, y_test)

min_val = min(min(y_pred_full), min(y_test))
max_val = max(max(y_pred_full), max(y_test))
plt.plot([min_val, max_val], [min_val, max_val], color="red")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.savefig("outputs/predicted_vs_actual.png")
plt.show()

# Interpretation:
# Points above the diagonal = underprediction (actual > predicted)
# Points below the diagonal = overprediction (predicted > actual)
