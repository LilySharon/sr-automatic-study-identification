# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:09:25 2023
updated on Thu Aug 1 12:127 2024

@author: Flavien
"""

import pandas as pd
import numpy as np
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
import pke
import spacy


def pke_keyphrase_extractor(model_name, df, df_column, top_n_list):
    """
    Extract keywords using specified keyword extraction algorithms from the pke library.

    Parameters:
    model_name (str): The name of the keyword extraction model (e.g., 'TfIdf', 'YAKE', 'TextRank', 'TopicRank').
    df (pd.DataFrame): DataFrame containing the text data.
    df_column (str): The name of the column in the DataFrame containing the text data.
    top_n_list (int): The number of top keyphrases to extract.

    Returns:
    list: A list of lists containing the extracted keywords for each document in the DataFrame.
    """
    nlp = spacy.load("en_core_web_md")

    # Initialize a keyphrase extraction model
    extractor = eval(f"pke.unsupervised.{model_name}()")
    
    # Loop through each row in the DataFrame and extract keyphrases
    keywords_list = []
    for i in range(len(df)):
        # Load the content of the document
        text = df.loc[i, df_column]
        doc = nlp(text)
        
        extractor.load_document(input=doc)  
        
        # Identify keyphrase candidates
        extractor.candidate_selection() 
        
        # Weight keyphrase candidates
        extractor.candidate_weighting() 
        
        # Select the n-best candidates as keyphrases
        keyphrases = extractor.get_n_best(n=top_n_list) 
        keyphrases_table = pd.DataFrame(keyphrases, columns=["keyword", "score"])
        keywords_list.append(list(keyphrases_table["keyword"]))
    
    return keywords_list


def create_pke_kw_cols(model_list, df, df_column, top_n=5):
    """
    Create a column for each keyword extraction algorithm in the model_list on the DataFrame.

    Parameters:
    model_list (list): List of keyword extraction models to use (e.g., ['TfIdf', 'YAKE', 'TextRank', 'TopicRank']).
    df (pd.DataFrame): DataFrame containing the text data.
    df_column (str): The name of the column in the DataFrame containing the text data.
    top_n (int): The number of top keyphrases to extract for each model.

    Returns:
    pd.DataFrame: DataFrame with new columns for each keyword extraction model.
    """
    for model_name in model_list:
        df[model_name.lower()] = pke_keyphrase_extractor(model_name, df, df_column, top_n)
    return df


def keybert_extractor(doc):
    """
    Extract keywords using the KeyBERT model.

    Parameters:
    doc (str): The document text from which to extract keywords.

    Returns:
    list: A list of extracted keywords.
    """
    kw_model = KeyBERT(model='all-mpnet-base-v2')
    keywords = kw_model.extract_keywords(docs=doc, vectorizer=KeyphraseCountVectorizer())
    keyword_table = pd.DataFrame(keywords, columns=["keyword", "score"])
    return list(keyword_table["keyword"])


def create_keybert_kw_cols(df, df_column):
    """
    Create a column for KeyBERT keyword extraction on the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data.
    df_column (str): The name of the column in the DataFrame containing the text data.

    Returns:
    pd.DataFrame: DataFrame with a new column for KeyBERT keyword extraction.
    """
    df['keybert'] = df[df_column].apply(lambda x: keybert_extractor(x))
    return df


class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    """
    Custom keyphrase extraction pipeline using a specified model for token classification.

    Attributes:
    model (str): The name of the model to use for keyphrase extraction.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, all_outputs):
        """
        Postprocess the outputs to extract unique keyphrases.

        Parameters:
        all_outputs (list): The outputs from the model.

        Returns:
        np.ndarray: An array of unique keyphrases extracted from the outputs.
        """
        results = super().postprocess(
            all_outputs=all_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])


# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
kbir_extractor = KeyphraseExtractionPipeline(model=model_name)


def create_kbir_kw_cols(df, df_column):
    """
    Create a column for KBIR keyword extraction on the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the text data.
    df_column (str): The name of the column in the DataFrame containing the text data.

    Returns:
    pd.DataFrame: DataFrame with a new column for KBIR keyword extraction.
    """
    df['kbir'] = df[df_column].apply(lambda x: [word.lower() for word in kbir_extractor(x)])
    return df


def extract_kw_cols(df, cols_list):
    """
    Extract keywords for each column in cols_list and combine the results into a set of unique keywords.

    Parameters:
    df (pd.DataFrame): DataFrame containing the keyword columns.
    cols_list (list): List of columns containing keywords (e.g., ['tfidf', 'yake', 'textrank', 'topicrank', 'keybert']).

    Returns:
    set: A set of unique keywords extracted from all specified columns.
    """
    list_term = []
    
    for col in cols_list:
        keywords_list = df[col].tolist()
        list_term.extend(keywords_list)

    def combine_kw_list(list_term):
        final_set = set()
        for lst in list_term:
            new_set = set(lst)
            final_set.update(new_set)
        return final_set
    
    return combine_kw_list(list_term)

def evaluate(top_N_keyphrases, references):
    """
    Evaluates the performance of extracted keyphrases against reference keyphrases using Precision, Recall, and F1-score.

    Parameters:
    top_N_keyphrases (list): List of top N extracted keyphrases.
    references (list): List of reference keyphrases.

    Returns:
    tuple: A tuple containing Precision (P), Recall (R), and F1-score (F).
    """
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases)
    R = len(set(top_N_keyphrases) & set(references)) / len(references)
    F = (2 * P * R) / (P + R) if (P + R) > 0 else 0 
    return (P, R, F)
