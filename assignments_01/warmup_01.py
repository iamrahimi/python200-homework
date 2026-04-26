import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

# ######################## 
# PANDAS QUESTION 1 ANSWER 
# ########################

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

# First 3 rows
print("First 3 Rows:")
print(df.head(3))

# Shape
print(f"\nShape: {df.shape}")
print(f"Num Rows: {len(df)}")


# ######################## 
# PANDAS QUESTION 2 ANSWER 

# Filter students who passed AND have grade > 80
filtered_df = df[(df['passed'] == True) & (df['grade'] > 80)]

print("Students who passed with grade > 80:")
print(filtered_df)


# ######################## 
# PANDAS QUESTION 3 ANSWER 


# Add a new column "grade_curved" = grade + 5
df['grade_curved'] = df['grade'] + 5

print("Updated DataFrame with 'grade_curved':")
print(df)

# ######################## 
# PANDAS QUESTION 4 ANSWER 


# Create a new column with uppercase names
df['name_upper'] = df['name'].str.upper()

# Print only "name" and "name_upper" columns
print(df[['name', 'name_upper']])

# ######################## 
# PANDAS QUESTION 5 ANSWER 


# Group by "city" and compute mean grade
city_mean_grade = df.groupby('city')['grade'].mean()

print("Mean grade by city:")
print(city_mean_grade)

# ######################## 
# PANDAS QUESTION 6 ANSWER 

# Replace "Austin" with "Houston" in the "city" column
df['city'] = df['city'].replace("Austin", "Houston")

# Print "name" and "city" columns to confirm
print(df[['name', 'city']])


# ######################## 
# PANDAS QUESTION 7 ANSWER 
# Sort by "grade" descending
sorted_df = df.sort_values(by='grade', ascending=False)

# Print top 3 rows
print("Top 3 students by grade:")
print(sorted_df.head(3))

# ######################## 
# NumPy QUESTION 1 ANSWER 

# Create a 1D array
arr = np.array([10, 20, 30, 40, 50])

# Print properties
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Number of dimensions (ndim): {arr.ndim}")


# ######################## 
# NumPy QUESTION 2 ANSWER 

# Create the 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Print the array
print("Array:")
print(arr)

print(f"\nShape: {arr.shape}")  
print(f"Size (total elements): {arr.size}")


# ######################## 
# NumPy QUESTION 3 ANSWER 

top_left = arr[:2, :2]  

print("Top-left 2x2 block:")
print(top_left)


# ######################## 
# NumPy QUESTION 4 ANSWER 
zeros_array = np.zeros((3, 4))
print("3x4 array of zeros:")
print(zeros_array)

# 2x5 array of ones
ones_array = np.ones((2, 5))
print("\n2x5 array of ones:")
print(ones_array)

# ######################## 
# NumPy QUESTION 5 ANSWER 

arr = np.arange(0, 50, 5)  
print("\nArray from np.arange(0,50,5):")
print(arr)
print(f"Shape: {arr.shape}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
print(f"Standard Deviation: {arr.std()}")

# ######################## 
# NumPy QUESTION 6 ANSWER 
rand_arr = np.random.normal(loc=0, scale=1, size=200)  
print("\nRandom array mean:", rand_arr.mean())
print("Random array standard deviation:", rand_arr.std())


# ######################## 
# Matplotlib QUESTION 1 ANSWER 

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y, marker='o', linestyle='-', color='blue')  

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

plt.show()


# ######################## 
# Matplotlib QUESTION 2 ANSWER 


subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
plt.bar(subjects, scores, color='skyblue')
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

plt.show()

# ######################## 
# Matplotlib QUESTION 3 ANSWER 

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')

plt.title("Scatter Plot of Two Datasets")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.legend()
plt.show()

# ######################## 
# Matplotlib QUESTION 4 ANSWER 

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x, y, marker='o', color='blue')
axes[0].set_title("Line Plot: X vs Y")
axes[0].set_xlabel("X-axis")
axes[0].set_ylabel("Y-axis")

axes[1].bar(subjects, scores, color='skyblue')
axes[1].set_title("Bar Plot: Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")


plt.tight_layout()
plt.show()

# ######################## 
# Descriptive QUESTION 1 ANSWER 

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data)

mean_val = arr.mean()
median_val = np.median(arr)
variance_val = arr.var()
std_dev_val = arr.std()

print(f"Mean: {mean_val}")
print(f"Median: {median_val}")
print(f"Variance: {variance_val}")
print(f"Standard Deviation: {std_dev_val}")

# ######################## 
# Descriptive QUESTION 1 ANSWER 

scores = np.random.normal(loc=65, scale=10, size=500)

plt.hist(scores, bins=20, color='skyblue', edgecolor='black')

plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")

plt.show()

# ######################## 
# Descriptive QUESTION 3 ANSWER 

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])

plt.title("Score Comparison")

plt.ylabel("Scores")
plt.xlabel("Groups")
plt.show()

# ######################## 
# Descriptive QUESTION 4 ANSWER 

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.xlabel("Distributions")
plt.show()


# ######################## 
# Descriptive QUESTION 5 ANSWER 

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mean1 = np.mean(data1)
median1 = np.median(data1)
mode1 = stats.mode(data1, keepdims=True).mode[0]

print("Data1:")
print(f"Mean: {mean1}")
print(f"Median: {median1}")
print(f"Mode: {mode1}\n")

mean2 = np.mean(data2)
median2 = np.median(data2)
mode2 = stats.mode(data2, keepdims=True).mode[0]

print("Data2:")
print(f"Mean: {mean2}")
print(f"Median: {median2}")
print(f"Mode: {mode2}")

# ######################## 
# Hypothesis QUESTION 1 ANSWER 

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# ######################## 
# Hypothesis QUESTION 2 ANSWER 
alpha = 0.05

if p_value < alpha:
    print("The result is statistically significant (reject the null hypothesis).")
else:
    print("The result is NOT statistically significant (fail to reject the null hypothesis).")



# ######################## 
# Hypothesis QUESTION 3 ANSWER 
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_value = stats.ttest_rel(after, before)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# ######################## 
# Hypothesis QUESTION 4 ANSWER 

scores = [72, 68, 75, 70, 69, 74, 71, 73]
benchmark = 70

t_stat, p_value = stats.ttest_1samp(scores, popmean=benchmark)

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


# ######################## 
# Hypothesis QUESTION 5 ANSWER 
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative='less')

print(f"One-tailed T-test p-value (group_a < group_b): {p_value}")


# ######################## 
# Hypothesis QUESTION 6 ANSWER 
print("Students in group_a scored lower on average than students in group_b. "
      "The difference is statistically significant, meaning it is very unlikely to have occurred by chance.")

# ######################## 
# Correlation Review

# 001 

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)

print("Correlation matrix:")
print(corr_matrix)

corr_coef = corr_matrix[0, 1]
print(f"\nPearson correlation coefficient between x and y: {corr_coef}")

# 002

# Data
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_coef, p_value = pearsonr(x, y)

print(f"Pearson correlation coefficient: {corr_coef}")
print(f"P-value: {p_value}")


# 003

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}

df = pd.DataFrame(people)
corr_matrix = df.corr()

print("Correlation matrix:")
print(corr_matrix)

# 004
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]


plt.scatter(x, y, color='red', marker='o')

plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")

plt.show()


# 005

people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}


df = pd.DataFrame(people)
corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")

plt.show()


#  ***********************
#  Pipelines      

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# Convert to pandas Series
def create_series(arr):
    return pd.Series(arr, name="values")

# Clean data by removing NaNs
def clean_data(series):
    return series.dropna()

# Summarize the cleaned data
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

# Data pipeline connecting the three functions
def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary


result = data_pipeline(arr)

for key, value in result.items():
    print(f"{key}: {value}")
