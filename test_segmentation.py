import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import air  # import your air.py script

# You might want to provide the same file path here:
FILE_PATH = "C:/Users/zahra/Downloads/AB_NYC_2019.csv"

@pytest.fixture
def cleaned_data():
    # Load and prepare the data exactly as in air.py but minimal for testing
    df = pd.read_csv(FILE_PATH)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df = df[(df['price'] > 0) & (df['price'] < 10000)]
    df = df[df['minimum_nights'] < 365]
    return df

def test_data_loaded():
    df = pd.read_csv(FILE_PATH)
    assert not df.empty
    assert 'price' in df.columns

def test_scaler_behavior():
    data = np.array([[1, 2], [3, 4]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    assert scaled.shape == data.shape

def test_kmeans_labels():
    # Test basic KMeans on small sample (like in your test)
    data = np.array([[0, 0], [1, 1], [10, 10]])
    model = KMeans(n_clusters=2, random_state=42)
    labels = model.fit_predict(data)
    assert len(labels) == 3
    assert set(labels).issubset({0, 1})

def test_pipeline_runs(cleaned_data):
    features = cleaned_data[['price', 'minimum_nights', 'number_of_reviews',
                             'reviews_per_month', 'availability_365']]
    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=4, random_state=42)
    cleaned_data['cluster'] = model.fit_predict(X)
    assert 'cluster' in cleaned_data.columns
    assert cleaned_data['cluster'].notnull().all()

def test_cluster_diversity(cleaned_data):
    features = cleaned_data[['price', 'minimum_nights', 'number_of_reviews',
                             'reviews_per_month', 'availability_365']]
    X = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=4, random_state=42)
    cleaned_data['cluster'] = model.fit_predict(X)
    assert cleaned_data['cluster'].nunique() >= 2

def test_column_consistency(cleaned_data):
    original_cols = cleaned_data.shape[1]
    processed = cleaned_data.copy()
    processed['new_column'] = processed['price'] * 0.1  # simulated change
    assert processed.shape[1] == original_cols + 1
