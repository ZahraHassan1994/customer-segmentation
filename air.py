import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import plotly.express as px

file_path = r"C:\Users\zahra\Downloads\airbab\AB_NYC_2019.csv"

df= pd.read_csv(file_path)
print(df.head())
print(df.columns.tolist())
df.info()
print(df.isnull().sum())
print(df.select_dtypes(include= 'object').head())
df['last_review']= pd.to_datetime(df['last_review'])
meanR= df['reviews_per_month'].mean()
df['reviews_per_month']= df['reviews_per_month'].fillna(meanR)
df= df[df['price']>0]
df.reset_index(drop=True, inplace= True)

total_listing= df.shape[0]
print(f"total_listing: {total_listing}")
print(f"ttl:",df.shape)
avg_price= df['price'].mean()
ravg_price= round(avg_price,2)
print(ravg_price)
common_room_type = df['room_type'].value_counts().idxmax()
print(f"common room", common_room_type)
df['room_type']= df['room_type'].astype('category')
print(df['room_type'].dtype)


#segmentation

df['reviews_per_month']= df['reviews_per_month'].fillna(0)
df= df[(df['price']>0) & (df['price']<10000)]
df= df[df['minimum_nights']<365]
#select features
featurs= df[['price','minimum_nights','number_of_reviews', 'reviews_per_month', 'availability_365']]
#standard feature
scaler= StandardScaler()
x_scaled= scaler.fit_transform(featurs)

#elbow
inertia=[]
K= range(1,11)
for k in K:
    Kmeans= KMeans(n_clusters=k, random_state=42)
    Kmeans.fit(x_scaled)
    inertia.append(Kmeans.inertia_)

plt.figure(figsize=(8,4))
plt.plot(K, inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method')
plt.grid(True)
plt.show()


for k in range(5,11):
    Kmeans= KMeans(n_clusters=k , random_state=42)
    labels= Kmeans.fit_predict(x_scaled)
    score= silhouette_score(x_scaled,labels)
    print(f"Silhouette Score for k={k}: {score:.4f}")

Kmeans= KMeans(n_clusters=7, random_state=42)
df['cluster']= Kmeans.fit_predict(x_scaled)

tsne= TSNE(n_components=2, perplexity=30, max_iter=300,random_state=42)
tsne_result = tsne.fit_transform(x_scaled)
df['tsne-2d-one'] = tsne_result[:, 0]
df['tsne-2d-two'] = tsne_result[:, 1]
fig = px.scatter(
    df,
    x='tsne-2d-one',
    y='tsne-2d-two',
    color='cluster',
    hover_data=['name', 'host_name', 'price', 'room_type', 'neighbourhood'],
    title="Customer Segmentation (Airbnb Listings) – KMeans + t-SNE"
)
fig.show()


