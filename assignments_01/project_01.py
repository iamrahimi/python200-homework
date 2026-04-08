from prefect import task, flow, get_run_logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pearsonr


# Task 1: Load Multiple Years of Data
@task(retries=3, retry_delay_seconds=2)
def load_and_merge_github_csv(urls):
    logger = get_run_logger()
    merged_df = pd.DataFrame()

    for url in urls:
        year = url.split('_')[-1].split('.')[0]
        try:
            df = pd.read_csv(url, on_bad_lines='skip', sep=None, engine='python')
            logger.info(f"Loaded {url} successfully")
        except Exception as e:
            logger.error(f"Failed to read {url}: {e}")
            continue

        # Add the year column
        df['year'] = int(year)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    output_path = "assignments_01/outputs/merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Merged dataset saved to {output_path} with shape {merged_df.shape}")

    return merged_df

# Flow
@flow
def happiness_pipeline_github():
    logger = get_run_logger()
    # List of raw GitHub CSV URLs
    urls = [
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2015.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2016.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2017.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2018.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2019.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2020.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2021.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2022.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2023.csv",
        "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/refs/heads/main/assignments/resources/happiness_project/world_happiness_2024.csv",
    ]

    merged_df = load_and_merge_github_csv(urls)
    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    df_merged = happiness_pipeline_github()


# --- Descriptive Statistics ---
logger = get_run_logger()

# Load the merged dataset
df = pd.read_csv("assignments_01/outputs/merged_happiness.csv")
df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

# Identify bad rows
bad_rows = df[df['happiness_score'].isna()]
logger.info(f"Rows with missing happiness_score:\n{bad_rows[['country', 'regional_indicator', 'happiness_score']]}")

# Overall descriptive stats
mean_score = df['happiness_score'].mean()
median_score = df['happiness_score'].median()
std_score = df['happiness_score'].std()

logger.info(f"Overall Happiness Score:\nMean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStandard Deviation: {std_score:.2f}")

# Mean happiness score grouped by year
year_group = df.groupby('year')['happiness_score'].mean()
logger.info(f"Mean Happiness Score by Year:\n{year_group}")

# Mean happiness score grouped by region
region_group = df.groupby('regional_indicator')['happiness_score'].mean()
logger.info("Mean Happiness Score by Region (highest to lowest):")
logger.info(region_group.sort_values(ascending=False))


# Task 3: Visual Exploration

@task
def create_visualizations(df: pd.DataFrame):
    logger = get_run_logger()
    output_dir = "assignments_01/outputs/"

    # Ensure numeric columns are proper types
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    #  Histogram of happiness scores
    plt.figure(figsize=(8,6))
    plt.hist(df['happiness_score'].dropna(), bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    hist_path = f"{output_dir}happiness_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    logger.info(f"Saved histogram: {hist_path}")

    #  Boxplot of happiness scores by year
    plt.figure(figsize=(10,6))
    sns.boxplot(x='year', y='happiness_score', data=df)
    plt.title("Happiness Scores by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    boxplot_path = f"{output_dir}happiness_by_year.png"
    plt.savefig(boxplot_path)
    plt.close()
    logger.info(f"Saved boxplot by year: {boxplot_path}")

    #  GDP per capita vs Happiness score
    plt.figure(figsize=(8,6))
    plt.scatter(df['gdp_per_capita'], df['happiness_score'], color='green', alpha=0.7)
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    scatter_path = f"{output_dir}gdp_vs_happiness.png"
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Saved scatter plot: {scatter_path}")

    #  Correlation heatmap of numeric columns
    plt.figure(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    heatmap_path = f"{output_dir}correlation_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved correlation heatmap: {heatmap_path}")


# Task 4: Hypothesis Testing
@task
def pandemic_impact_tests(df: pd.DataFrame):
    logger = get_run_logger()
    
    # --- Independent samples t-test: 2019 vs 2020 ---
    df_2019 = df[df['year'] == 2019]['happiness_score'].dropna()
    df_2020 = df[df['year'] == 2020]['happiness_score'].dropna()

    t_stat, p_value = ttest_ind(df_2019, df_2020, equal_var=False)
    mean_2019 = df_2019.mean()
    mean_2020 = df_2020.mean()

    logger.info("=== Pandemic Impact Test: 2019 vs 2020 ===")
    logger.info(f"Mean Happiness 2019: {mean_2019:.2f}")
    logger.info(f"Mean Happiness 2020: {mean_2020:.2f}")
    logger.info(f"T-statistic: {t_stat:.3f}")
    logger.info(f"P-value: {p_value:.5f}")

    if p_value < 0.05:
        logger.info("Interpretation: Happiness scores in 2020 were significantly different from 2019. "
                    "This suggests the pandemic may have affected global happiness.")
    else:
        logger.info("Interpretation: No significant difference in happiness scores between 2019 and 2020. "
                    "The pandemic does not appear to have had a detectable effect in this dataset.")

    df_we = df[df['regional_indicator'] == 'Western Europe']['happiness_score'].dropna()
    df_ssa = df[df['regional_indicator'] == 'Sub-Saharan Africa']['happiness_score'].dropna()

    t_stat_r, p_value_r = ttest_ind(df_we, df_ssa, equal_var=False)
    mean_we = df_we.mean()
    mean_ssa = df_ssa.mean()

    logger.info("=== Regional Comparison: Western Europe vs Sub-Saharan Africa ===")
    logger.info(f"Mean Happiness Western Europe: {mean_we:.2f}")
    logger.info(f"Mean Happiness Sub-Saharan Africa: {mean_ssa:.2f}")
    logger.info(f"T-statistic: {t_stat_r:.3f}")
    logger.info(f"P-value: {p_value_r:.5f}")

    if p_value_r < 0.05:
        logger.info("Interpretation: Western Europe has significantly higher happiness scores than Sub-Saharan Africa, "
                    "consistent with regional trends seen in the descriptive statistics.")
    else:
        logger.info("Interpretation: No significant difference between these regions, which is unexpected given prior data.")


# Task 5: Correlation and Multiple Comparisons


@task
def correlation_analysis(df):
    logger = get_run_logger()

    # Identify numeric explanatory variables (exclude happiness_score itself)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if 'happiness_score' in numeric_cols:
        numeric_cols.remove('happiness_score')

    logger.info(f"Numeric variables to test: {numeric_cols}")

    # Store results
    results = []

    for col in numeric_cols:
        # Drop NA for pairwise correlation
        valid = df[['happiness_score', col]].dropna()
        if len(valid) < 2:
            logger.warning(f"Not enough data to compute correlation for {col}")
            continue

        corr_coef, p_val = pearsonr(valid['happiness_score'], valid[col])
        results.append((col, corr_coef, p_val))
        logger.info(f"Correlation with {col}: r = {corr_coef:.3f}, p = {p_val:.5f}")

    # Bonferroni correction
    number_of_tests = len(results)
    adjusted_alpha = 0.05 / number_of_tests if number_of_tests > 0 else 0.05
    logger.info(f"Number of tests: {number_of_tests}, Adjusted alpha (Bonferroni): {adjusted_alpha:.5f}")

    # Log significance
    for col, corr_coef, p_val in results:
        sig_orig = p_val < 0.05
        sig_adj = p_val < adjusted_alpha
        logger.info(
            f"{col}: r={corr_coef:.3f}, p={p_val:.5f}, "
            f"Significant at alpha=0.05? {sig_orig}, "
            f"Significant after Bonferroni? {sig_adj}"
        )

@flow
def happiness_correlation_pipeline():
    logger = get_run_logger()

    # Load the merged dataset
    df = pd.read_csv("assignments_01/outputs/merged_happiness.csv")
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

    logger.info("Starting correlation analysis...")
    correlation_analysis(df)
    logger.info("Correlation analysis completed.")

if __name__ == "__main__":
    happiness_correlation_pipeline()


# Task 6: Summary Report
@task
def generate_report(df, correlations):
    logger = get_run_logger()

    # Total number of countries and years
    total_countries = df['country'].nunique()
    total_years = df['year'].nunique()
    logger.info(f"Total countries: {total_countries}")
    logger.info(f"Total years: {total_years}")

    # Top 3 and bottom 3 regions by mean happiness score
    region_means = df.groupby('regional_indicator')['happiness_score'].mean()
    logger.info(f"Top 3 regions by mean happiness score:\n{region_means.sort_values(ascending=False).head(3)}")
    logger.info(f"Bottom 3 regions by mean happiness score:\n{region_means.sort_values(ascending=True).head(3)}")

    # Pre/post-2020 t-test
    pre_2020 = df[df['year'] < 2020]['happiness_score'].dropna()
    post_2020 = df[df['year'] >= 2020]['happiness_score'].dropna()
    t_stat, p_val = ttest_ind(pre_2020, post_2020, equal_var=False)

    if p_val < 0.05:
        logger.info(f"Happiness scores changed significantly after 2020 (t={t_stat:.3f}, p={p_val:.5f})")
    else:
        logger.info(f"No significant change in happiness after 2020 (t={t_stat:.3f}, p={p_val:.5f})")

    # Variable most strongly correlated with happiness (after Bonferroni correction)
    significant_corrs = {k: v for k, v in correlations.items() if v['significant_bonf']}
    if significant_corrs:
        strongest_var = max(significant_corrs, key=lambda k: abs(significant_corrs[k]['r']))
        strongest_r = significant_corrs[strongest_var]['r']
        logger.info(f"Variable most strongly correlated with happiness (after Bonferroni correction): '{strongest_var}' with r={strongest_r:.3f}")
    else:
        logger.info("No variables remain significant after Bonferroni correction.")

# -----------------------------
# Main flow
# -----------------------------
@flow
def happiness_pipeline():
    # Example: load your merged dataset first
    df = pd.read_csv("assignments_01/outputs/merged_happiness.csv")
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    df['happiness_score'] = pd.to_numeric(df['happiness_score'], errors='coerce')

    # Compute correlations
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    numeric_cols.remove('happiness_score')
    correlations = {}
    adjusted_alpha = 0.05 / max(len(numeric_cols), 1)
    for col in numeric_cols:
        valid = df[['happiness_score', col]].dropna()
        if len(valid) < 2:
            continue
        r, p = pearsonr(valid['happiness_score'], valid[col])
        correlations[col] = {'r': r, 'p': p, 'significant': p < 0.05, 'significant_bonf': p < adjusted_alpha}

    # Generate human-readable report
    generate_report(df, correlations)

if __name__ == "__main__":
    happiness_pipeline()
