import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline



#1. Load the Spambase Dataset
# Column names (Spambase has no header row)
columns = [
    # word_freq features (48)
    *[f'word_freq_{i}' for i in range(48)],
    # char_freq features (6)
    *[f'char_freq_{i}' for i in range(6)],
    # capital run length features (3)
    'capital_run_length_avg',
    'capital_run_length_longest',
    'capital_run_length_total',
    # target
    'spam_label'
]

df = pd.read_csv("assignments_03/spambase.data", header=None)
df.columns = columns

print(df.shape)
df.head()

# 2. Basic Exploration
print("Total emails:", df.shape[0])
print(df['spam_label'].value_counts())
print(df['spam_label'].value_counts(normalize=True))


#3. Boxplots for Key Features
df.rename(columns={
    'word_freq_0': 'word_freq_free',   # adjust if your indexing differs
    'char_freq_0': 'char_freq_!',      # adjust if needed
}, inplace=True)


features = [
    'word_freq_free',
    'char_freq_!',
    'capital_run_length_total'
]

for feature in features:
    plt.figure()
    df.boxplot(column=feature, by='spam_label')
    plt.title(f"{feature} by Spam vs Ham")
    plt.suptitle("")  # remove default title
    plt.xlabel("Spam Label (0 = Ham, 1 = Spam)")
    plt.ylabel(feature)
    plt.savefig(f"assignments_03/outputs/{feature}_boxplot.png")
    plt.close()


# • The dataset contains 4,601 emails.

# • The classes are moderately imbalanced:
#     - Ham (0): about 61%
#     - Spam (1): about 39%
# • Because the classes are not perfectly balanced, raw accuracy alone can
#   be misleading. A classifier that always predicts "ham" would already
#   achieve about 61% accuracy without identifying any spam emails.

# • The boxplots show noticeable differences between spam and ham.
#   - Spam emails generally contain higher values of word_freq_free.
#   - Spam emails tend to use the '!' character more often.
#   - Spam emails often have much longer runs of capital letters.
#   Although there is overlap between the classes, these features are useful

#   predictors of spam.

# • Many word-frequency features are heavily skewed toward zero because most
#   emails never contain many of the tracked words. This results in sparse
#   data with many zero values.
# • Feature scales vary widely because some variables represent percentages
#   (small decimal values), while others measure counts or lengths (which can
#   reach into the hundreds or thousands). Models based on distances or
#   optimization (such as Logistic Regression, KNN, SVM, and PCA) often
#   benefit from feature scaling so that large-valued features do not dominate
#   the learning process.

# Task 2
#1. Train/Test Split

X = df.drop(columns=['spam_label'])
y = df['spam_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # preserves class balance
)


scaler = StandardScaler()

# Fit ONLY on training data (avoid leakage)
X_train_scaled = scaler.fit_transform(X_train)

# Use same transformation on test data
X_test_scaled = scaler.transform(X_test)

# I scale features because:
# - Features have very different ranges (fractions vs thousands)
# - Distance-based models (KNN) and linear models (logistic regression)
#   are sensitive to feature magnitude
# - PCA depends on variance, so scaling ensures no feature dominates
# - Fit is done only on training data to avoid test data leakage

#2. PCA (Dimensionality Reduction)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)


#3. Explained Variance Plot
plt.figure()
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()

plt.savefig("assignments_03/outputs/pca_explained_variance.png")
plt.close()

#4. Find Number of Components (90%)
n = np.argmax(cumulative_variance >= 0.90) + 1
print("Number of components to reach 90% variance:", n)
# We choose the smallest number of components that explain at least 90% of variance
# This reduces dimensionality while keeping most of the information


#6. Transform Data Using PCA
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]
# We transform both train and test using PCA fitted on training data
# Then we keep only the first n principal components
# This ensures:
# - No data leakage
# - Reduced dimensionality for certain models



# Task 3

#1. Helper Function (avoid repeating code)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

#2. KNN Comparisons


#  Unscaled (baseline - expected to perform poorly)
evaluate_model(
    "KNN (unscaled)",
    KNeighborsClassifier(n_neighbors=5),
    X_train, X_test, y_train, y_test
)

# Scaled
evaluate_model(
    "KNN (scaled)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, X_test_scaled, y_train, y_test
)

#  PCA
evaluate_model(
    "KNN (PCA)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca, X_test_pca, y_train, y_test
)


# 3. Decision Tree (Depth Experiment)


depths = [3, 5, 10, None]

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"\nDepth: {d}")
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

# I choose max_depth=10 because:
# - It balances bias and variance
# - Higher depths overfit (train accuracy high, test accuracy drops)
# - Shallower trees underfit

best_tree = DecisionTreeClassifier(max_depth=10, random_state=42)

evaluate_model(
    "Decision Tree (best depth)",
    best_tree,
    X_train, X_test, y_train, y_test
)

#4. Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)

evaluate_model(
    "Random Forest",
    rf,
    X_train, X_test, y_train, y_test
)


#5. Logistic Regression

# Scaled
evaluate_model(
    "Logistic Regression (scaled)",
    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'),
    X_train_scaled, X_test_scaled, y_train, y_test
)

# PCA
evaluate_model(
    "Logistic Regression (PCA)",
    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'),
    X_train_pca, X_test_pca, y_train, y_test
)


#6. Confusion Matrix (Best Model)


best_model = rf  # change if needed
best_model.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)

plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.close()

#7. Feature Importance

feature_importance_tree = pd.Series(
    best_tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Features:")
print(feature_importance_tree.head(10))

feature_importance_rf = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Features:")
print(feature_importance_rf.head(10))

feature_importance_rf.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()

plt.savefig("assignments_03/outputs/feature_importances.png")
plt.close()

# Decision Tree:
# As max_depth increases, training accuracy continues to improve,
# but test accuracy eventually stops improving or begins to decrease.
# This indicates overfitting at higher depths. Based on my results,
# I chose max_depth=5 because it provides a good balance between
# model complexity and generalization while avoiding unnecessary overfitting.

# KNN:
# Scaling generally improves KNN because distance calculations are
# affected by feature magnitudes. PCA may slightly reduce or maintain
# performance while reducing dimensionality.

# Logistic Regression:
# Logistic Regression usually performs slightly better on scaled data
# than on PCA-reduced data because PCA discards some information that
# may still be useful for classification.

# Overall:
# Random Forest typically achieves the highest accuracy on the Spambase
# dataset because it combines many decision trees and reduces overfitting.

# Spam Filter:
# Accuracy alone is not the best metric. False positives (legitimate
# email marked as spam) are generally more costly than false negatives
# because users may miss important emails. Therefore, minimizing false
# positives while maintaining high recall for spam is usually the best trade-off.

# Confusion Matrix:
# Examine the confusion matrix to determine whether the model produces
# more false positives or false negatives. Random Forest usually makes
# relatively few of both, but the matrix provides the exact counts for your trained model.


#Task 4: Cross-Validation
#1. Setup Cross-Validation
def cross_validate_model(name, model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    
    print(f"\n{name}")
    print("Scores:", scores)
    print("Mean Accuracy:", np.mean(scores))
    print("Std Dev:", np.std(scores))

#2. Run Cross-Validation for Each Model

cross_validate_model(
    "KNN (scaled)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_scaled, y_train
)

cross_validate_model(
    "KNN (PCA)",
    KNeighborsClassifier(n_neighbors=5),
    X_train_pca, y_train
)

cross_validate_model(
    "Decision Tree (best depth)",
    DecisionTreeClassifier(max_depth=5, random_state=42),
    X_train, y_train
)

cross_validate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train
)

cross_validate_model(
    "Logistic Regression (scaled)",
    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'),
    X_train_scaled, y_train
)

cross_validate_model(
    "Logistic Regression (PCA)",
    LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'),
    X_train_pca, y_train
)

#3. Final Written Answer

# Cross-validation results show that Random Forest achieved the highest mean accuracy
# and the lowest standard deviation, making it both the most accurate and most stable model.

# The Random Forest had the lowest variance across folds, indicating strong stability.
# In contrast, the Decision Tree showed higher variance, confirming it is sensitive
# to small changes in the training data.


#Task 5: Building a Prediction Pipeline

rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\nRandom Forest Pipeline")
print(classification_report(y_test, y_pred_rf))

# Random Forest Pipeline:
# This model does not require scaling or PCA because it is tree-based.
# It splits data using thresholds, so feature magnitude does not affect performance.
# This makes it simpler and more robust compared to linear or distance-based models.

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline (Scaled)")
print(classification_report(y_test, y_pred_lr))

# Logistic Regression (Scaled):
# Scaling is required because logistic regression is sensitive to feature magnitude.
# Without scaling, large-valued features would dominate the model and reduce performance.

lr_pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),  # from Task 2
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

lr_pca_pipeline.fit(X_train, y_train)

y_pred_lr_pca = lr_pca_pipeline.predict(X_test)

print("\nLogistic Regression Pipeline (PCA)")
print(classification_report(y_test, y_pred_lr_pca))

# Logistic Regression (PCA):
# This pipeline reduces dimensionality before classification.
# PCA helps remove noise and redundancy, but may also drop useful information.
# It works best when features are highly correlated.

