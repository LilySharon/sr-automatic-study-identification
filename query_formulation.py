# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:32:48 2024
updated on Thu Aug 1 12:12 2024

@author: Flavien 
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import nmslib
import requests
import os


class ProcessData:
    """
    Class to process and load data from a CSV file.
    """

    def __init__(self, file_path):
        """
        Initializes the ProcessData object.

        Parameters:
        file_path (str): The file path to the CSV data file.
        """
        self.file_path = file_path
        self.df_ai = self.data_load()
        self.docs = self.df_ai["text"].tolist()

    def data_load(self):
        """
        Loads data from a CSV file, processes it, and returns a DataFrame.

        Returns:
        DataFrame: Processed DataFrame containing 'title', 'abstract', and 'text' columns.
        """
        df_raw = pd.read_csv(self.file_path, encoding='ISO-8859-1')
        dataset = df_raw.copy()

        dataset["abstract"] = dataset["abstract"].fillna("")
        text_columns = ["title", "abstract"]
        for col in text_columns:
            dataset[col] = dataset[col].astype(str)
        dataset["text"] = dataset[text_columns].apply(lambda x: " ".join(x), axis=1)
        return dataset[["title", "abstract", "text"]]

    def random_split_data(self):
        """
        Splits the DataFrame into four equal parts after shuffling it.

        Returns:
        list: A list of DataFrames, each containing a part of the original data.
        """
        num_parts = 4
        df_ai_shuffled = self.df_ai.sample(frac=1).reset_index(drop=True)
        observations_per_part = len(df_ai_shuffled) // num_parts
        split_df = [df_ai_shuffled.iloc[i:i + observations_per_part] for i in range(0, len(df_ai_shuffled), observations_per_part)]
        return split_df


def generate_synonyms(updated_terms, model_name='all-MiniLM-L6-v2', num_neighbors=2):
    """
    Generates synonyms for a list of terms using SentenceTransformer embeddings and NMSLIB.

    Parameters:
    updated_terms (list): List of terms to generate synonyms for.
    model_name (str): Name of the SentenceTransformer model to use.
    num_neighbors (int): Number of nearest neighbors to consider for generating synonyms.

    Returns:
    dict: A dictionary where keys are original terms and values are lists of synonyms.
    """
    model = SentenceTransformer(model_name)
    term_vectors = model.encode(updated_terms)
    term_vectors_normalized = term_vectors / np.linalg.norm(term_vectors, axis=1, keepdims=True)
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(term_vectors_normalized)
    index.createIndex(print_progress=True)

    synonyms = {}
    for i, term in enumerate(updated_terms):
        ids, _ = index.knnQuery(term_vectors_normalized[i], k=num_neighbors + 1)
        synonyms[term] = [updated_terms[idx] for idx in ids[1:]]
    return synonyms


class QueryFormulation:
    """
    Class to formulate and execute search queries based on clustered keywords.
    """

    def __init__(self, keywords, api_key, embedding_model_name="all-MiniLM-L6-v2", num_clusters=3):
        """
        Initializes the QueryFormulation object.

        Parameters:
        keywords (list): List of keywords for clustering and query formulation.
        api_key (str): API key for accessing external search APIs.
        embedding_model_name (str): Name of the SentenceTransformer model to use.
        num_clusters (int): Number of clusters for keyword clustering.
        """
        self.keywords = keywords
        self.api_key = api_key
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.doc_embeddings = self.embedding_model.encode(self.keywords)
        self.num_clusters = num_clusters
        self.clustered_terms = self.cluster_terms()

    def cluster_terms(self):
        """
        Clusters the keywords using KMeans clustering.

        Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of keywords in each cluster.
        """
        kmeans_model = KMeans(n_clusters=self.num_clusters)
        kmeans_model.fit(self.doc_embeddings)
        cluster_assignment = kmeans_model.labels_

        clustered_keywords = {}
        for keyword_id, cluster_id in enumerate(cluster_assignment):
            if cluster_id not in clustered_keywords:
                clustered_keywords[cluster_id] = []
            clustered_keywords[cluster_id].append(self.keywords[keyword_id])

        return clustered_keywords

    @classmethod
    def formulate_query(cls, clustered_terms):
        """
        Formulates a query string from clustered terms.

        Parameters:
        clustered_terms (dict): A dictionary where keys are cluster IDs and values are lists of keywords.

        Returns:
        str: A formulated query string.
        """
        joined_lists = []
        for terms_list in clustered_terms.values():
            joined_list = ' OR '.join(['"' + term + '"' for term in terms_list])
            joined_lists.append(joined_list)
        final_query = ' AND '.join(joined_lists)
        return f'TITLE-ABS-KEY({final_query})'

    @classmethod
    def integrate_synonyms(cls, clustered_terms, synonyms):
        """
        Integrates synonyms into the clustered terms.

        Parameters:
        clustered_terms (dict): A dictionary where keys are cluster IDs and values are lists of keywords.
        synonyms (dict): A dictionary where keys are original terms and values are lists of synonyms.

        Returns:
        dict: Updated clustered terms including synonyms.
        """
        for cluster_id in clustered_terms:
            clustered_terms[cluster_id].extend(synonyms.get(clustered_terms[cluster_id][0], []))
        return clustered_terms

    def scopus_search(self, output_dir='scopus_results', query_id=None):
        """
        Searches Scopus using the formulated query and saves results to a CSV file.

        Parameters:
        output_dir (str): Directory to save the results.
        query_id (str or None): Optional query ID for naming the output file.

        Returns:
        None
        """
        query = self.formulate_query(self.clustered_terms)
        base_url = 'https://api.elsevier.com/content/search/scopus'
        params = {
            'query': query,
            'apiKey': self.api_key,
            'count': 25,
            'start': 0,
            'httpAccept': 'application/json'
        }

        results = []

        while True:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                entries = data.get('search-results', {}).get('entry', [])
                if not entries:
                    break

                for entry in entries:
                    title = entry.get('dc:title')
                    authors = entry.get('dc:creator')
                    results.append({
                        'title': title,
                        'authors': authors
                    })
                params['start'] += params['count']
            else:
                print(f"Error: {response.status_code} - {response.text}")
                break

        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'scopus_results_{query_id if query_id else "query"}.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)


class RecallPrecisionFBeta:
    """
    Class to calculate Recall, Precision, and F-beta score for search results.
    """

    def __init__(self, gs_search, scopus):
        """
        Initializes the RecallPrecisionFBeta object.

        Parameters:
        gs_search (DataFrame): DataFrame containing ground truth search results.
        scopus (DataFrame): DataFrame containing Scopus search results.
        """
        self.gs_search = gs_search
        self.scopus = scopus
        self.scopus_set = set(scopus.title)
        self.gs_search_set = set(gs_search.title)
        self.R_found = len(self.scopus_set.intersection(self.gs_search_set))
        self.R_total = len(self.gs_search_set)

    def recall(self):
        """
        Calculates the recall score.

        Returns:
        float: The recall score.
        """
        if self.R_total == 0:
            return 0
        return self.R_found / self.R_total

    def precision(self):
        """
        Calculates the precision score.

        Returns:
        float: The precision score.
        """
        N_total = len(self.scopus_set)
        if N_total == 0:
            return 0
        return round(self.R_found / N_total, 4)

    def f_beta_score(self, beta=1):
        """
        Calculates the F-beta score.

        Parameters:
        beta (int): The beta value for the F-beta score calculation.

        Returns:
        float: The F-beta score.
        """
        recall = self.recall()
        precision = self.precision()

        if precision + recall == 0:
            return 0

        return round((1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall), 4)

