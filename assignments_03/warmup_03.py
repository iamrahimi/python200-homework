import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.multiclass import OneVsRestClassifier

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# ---- Preprocessing ---- 

#Q1 

# Split into training and test sets (80/20, stratified)
# Split data (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q2 

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use same scaler to transform test data
X_test_scaled = scaler.transform(X_test)

# Print mean of each column in scaled training data
print("Means of X_train_scaled columns:", np.mean(X_train_scaled, axis=0))


# --- KNN ---
# Q1: KNN without scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("\nKNN Q1: Unscaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Q2: KNN with scaling
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN Q2: Scaled Data")
print("Accuracy:", accuracy_score(y_test, y_pred_scaled))


# Q3: Cross-validation (k=5) on unscaled training data
knn_cv = KNeighborsClassifier(n_neighbors=5)

cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

print("\nKNN Q3: Cross-validation scores (k=5)")
print("Fold scores:", cv_scores)
print("Mean:", np.mean(cv_scores))
print("Standard deviation:", np.std(cv_scores))


# Q4: Try different k values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("\nKNN Q4: Different k values (CV mean accuracy)")

best_k = None
best_score = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    mean_score = np.mean(scores)

    print(f"k={k}, Mean CV Accuracy={mean_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print(f"\nBest k: {best_k} with score: {best_score:.4f}")

# --- Classifier Evaluation ---
# Q1: Confusion Matrix for KNN (from Q1: unscaled)


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()


plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved to outputs/knn_confusion_matrix.png")

# --- Decision Trees ---
# Q1: Decision Tree model

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# --- Logistic Regression and Regularization ---
# Q1: Different C values

C_values = [0.01, 1.0, 100]

print("\nLogistic Regression (Coefficient Magnitudes)")


for C in C_values:
    lr = OneVsRestClassifier(
        LogisticRegression(C=C, max_iter=1000, solver='liblinear')
    )
    lr.fit(X_train_scaled, y_train)

    coef_size = sum(np.abs(model.coef_).sum() for model in lr.estimators_)

    print(f"C={C}, Total Coefficient Magnitude={coef_size:.4f}")

# --- PCA ---

digits = load_digits()
X_digits = digits.data    # (1797, 64)
y_digits = digits.target
images = digits.images    # (1797, 8, 8)

# Q1: Shapes and sample images
print("\nPCA Q1: Shapes")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

# Plot one example of each digit (0–9)
fig, axes = plt.subplots(1, 10, figsize=(10, 2))

for digit in range(10):
    # find first index of this digit
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap='gray_r')
    axes[digit].set_title(str(digit))
    axes[digit].axis('off')

plt.savefig("assignments_03/outputs/sample_digits.png")
plt.close()

print("Sample digits figure saved to outputs/sample_digits.png")


# Q2: PCA projection
pca = PCA()
pca.fit(X_digits)

scores = pca.transform(X_digits)

# Scatter plot using first 2 principal components
plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    scores[:, 0],
    scores[:, 1],
    c=y_digits,
    cmap='tab10',
    s=10
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection of Digits")

plt.colorbar(scatter, label='Digit')

plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.close()

print("PCA plot saved to outputs/pca_2d_projection.png")

# --- Q3: Explained Variance ---

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(6, 4))
plt.plot(explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.grid(True)

plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.close()

print("\nPCA variance plot saved to outputs/pca_variance_explained.png")


# --- Q4: Reconstruction ---

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction += scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


# Components to test
n_values = [2, 5, 15, 40]
num_digits = 5

fig, axes = plt.subplots(len(n_values) + 1, num_digits, figsize=(10, 8))

# --- Original row ---
for i in range(num_digits):
    axes[0, i].imshow(images[i], cmap='gray_r')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("Original")

# --- Reconstructions ---
for row, n in enumerate(n_values):
    for col in range(num_digits):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row + 1, col].imshow(recon, cmap='gray_r')
        axes[row + 1, col].axis('off')

        if col == 0:
            axes[row + 1, col].set_ylabel(f"n={n}")

plt.tight_layout()

plt.savefig("assignments_03/outputs/pca_reconstructions.png")
plt.close()

print("Reconstruction figure saved to outputs/pca_reconstructions.png")

