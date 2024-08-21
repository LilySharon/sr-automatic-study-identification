# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:28:15 2024

@author: fowo0001
"""
import pandas as pd
import query_formulation as qf
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

def beta_fscore(data, beta):
    """
    Calculate recall, precision, F-1 and beta score at various levels (0.5, 1, or 2) 
    beta=1 is F1-score else is beta score.

    Parameters:
    data (list): List of dataset file paths.
    beta (float): The beta value to use in the F-beta score calculation.

    Returns:
    pd.DataFrame: DataFrame containing recall, precision, and F1 score for each dataset.
    """
    # List of dataset paths
    dsl = data

    # Extract dataset names from paths
    dataset_names = [file_path.split('/')[-1].split('.')[0] for file_path in dsl]

    # Initialize an empty dictionary to store the results
    result_dict = {}

    # Iterate over each dataset path and name
    for file_path, dataset_name in zip(dsl, dataset_names):
        # Read the dataset
        scopus_df = pd.read_csv(file_path)

        # Calculate recall, precision, and F1-score
        R = qf.RecallPrecisionFBeta(gs_search, scopus_df)

        # Store the evaluation metrics in the dictionary
        result_dict[dataset_name] = [R.recall(), R.precision(), R.f_beta_score(beta)]

    # Convert the dictionary to a DataFrame
    results = pd.DataFrame.from_dict(result_dict, orient='index', columns=['Recall', 'Precision', 'F1_Score'])
    return results

def split_dataset_string(dataset_string):
    """
    Splits a dataset string into the base part and a list of algorithms.

    Parameters:
    dataset_string (str): The input string to split.

    Returns:
    tuple: A tuple containing the base dataset and a comma-separated string of algorithms.
    """
    parts = dataset_string.split('_')
    base = parts[0] + '_' + parts[1]  # "Dataset_1"
    algorithms = ', '.join(parts[2:])  # Join the remaining parts
    return base, algorithms

def split_dataset_column(df, column_name):
    """
    Splits the dataset column into two separate columns: 'dataset' and 'algorithms'.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to split.

    Returns:
    pd.DataFrame: A new DataFrame with 'dataset' and 'algorithms' columns.
    """
    # Create two new columns by splitting the existing column
    df[['Dataset', 'Algorithms']] = df[column_name].apply(lambda x: pd.Series(split_dataset_string(x)))
    return df

def check_normality_and_compare(expert_scores, random_scores, alpha=0.05):
    """
    Perform normality tests on two datasets and choose the appropriate statistical test to compare their means.
    
    Parameters:
    expert_scores (array-like): F1 scores for expert queries.
    random_scores (array-like): F1 scores for random queries.
    alpha (float): Significance level to determine normality and homogeneity of variances.
    
    Returns:
    dict: Dictionary containing the normality test results, homogeneity test results, 
          the chosen test, test statistic, and p-value.
    """
    # Check normality for expert scores
    _, p_expert_normality = shapiro(expert_scores)
    is_normal_expert = p_expert_normality > alpha
    
    # Check normality for random scores
    _, p_random_normality = shapiro(random_scores)
    is_normal_random = p_random_normality > alpha
    
    # Check homogeneity of variances
    _, p_levene = levene(expert_scores, random_scores)
    is_homogeneous = p_levene > alpha
    
    # Choose the appropriate test
    if is_normal_expert and is_normal_random and is_homogeneous:
        # Use Independent Samples t-test
        test_name = "Independent Samples t-test"
        stat, p_value = ttest_ind(expert_scores, random_scores)
    else:
        # Use Mann-Whitney U test
        test_name = "Mann-Whitney U test"
        stat, p_value = mannwhitneyu(expert_scores, random_scores)
    
    return {
        "normality_expert": is_normal_expert,
        "normality_random": is_normal_random,
        "homogeneity": is_homogeneous,
        "test": test_name,
        "statistic": stat,
        "p_value": p_value
    }
