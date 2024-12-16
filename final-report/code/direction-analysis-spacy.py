import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE  # Import t-SNE
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_md")

# Create DataFrame
# df = pd.DataFrame(data)
df = pd.read_csv('company_keywords.csv')

# Function to get the average vector of a list of words using Spacy
def get_average_vector(keywords, nlp):
    vectors = [nlp(word).vector for word in keywords if nlp(word).has_vector]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None
    
df['company_directions'] = df['company_directions'].apply(lambda x: x.replace(',',''))
df['company_directions'] = df['company_directions'].apply(lambda x: x.split())

# Apply the function to get company_vectors
df['company_vector'] = df['company_directions'].apply(lambda x: get_average_vector(x, nlp))

# Drop rows where no valid company_vector could be generated
df.dropna(subset=['company_vector'], inplace=True)

# Convert company_vectors into a matrix for clustering (each row is a company's vector)
X = np.array(df['company_vector'].tolist())

k_values = range(2, len(df['company_directions']) - 1)
inertia = []
silhouette_scores = []

# Code for trying to find good numbers of clusters
# Try different values of k and calculate silhouette scores
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)  # Sum of squared distances to centroids
    # Calculate silhouette score for each k
    score = silhouette_score(X, kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.xticks(k_values[0::10])
plt.grid(True)
# Plot silhouette scores for different k values
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='g')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values[0::10])
plt.grid(True)

plt.tight_layout()
plt.show()
plt.savefig('output-cluster-scores-spacy.png')

# Determine the best k (the one with the highest silhouette score)
# best_k = k_values[np.argmax(silhouette_scores)]
# print(f"The best value for k is {best_k} with a silhouette score of {max(silhouette_scores)}")

best_k = 25

# Perform clustering with the best k
best_kmeans = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = best_kmeans.fit_predict(X)

# Perform t-SNE to reduce dimensionality for visualization (to 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Cluster'], cmap='viridis', marker='o')
plt.title(f"Company Clusters (k={best_k}) Visualized using t-SNE")
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Annotate points with company names
for i, txt in enumerate(df['company_name']):
    plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=7)

# Display the plot
plt.colorbar(label='Cluster')
plt.show()
plt.savefig('output-clustering-spacy.png')

# View the companies and their corresponding clusters
# print(df[['company_name', 'company_directions','Cluster']])
df.to_csv('output-cluster.csv')

# Find the centroids (cluster centers)
centroids = best_kmeans.cluster_centers_

# Find the most influential words for each cluster (using cosine similarity)
word_list = list(set([word for sublist in df['company_directions'] for word in sublist]))  # All unique words in the dataset
word_vectors = np.array([nlp(word).vector for word in word_list])

# print(word_list)

# Find the cosine similarity between each centroid and each word
cosine_similarities = cosine_similarity(centroids, word_vectors)

# Print the most influential words for each cluster
for i in range(best_k):
    print(f"\nCluster {i} - Most Influential Words:")
    # Get indices of words most similar to the centroid
    most_similar_indices = np.argsort(cosine_similarities[i])[::-1][:5]  # Top 5 most similar words
    for index in most_similar_indices:
        print(f"    {word_list[index]} (Similarity: {cosine_similarities[i][index]:.2f})")

for i in range(best_k):
    print(f"Cluster {i}:")
    print(df[df['Cluster'] == i])