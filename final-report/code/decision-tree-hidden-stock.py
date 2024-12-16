import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import spacy
import yfinance as yf
import datetime
import sqlite3
import json
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

tickers = [
    'NVDA', 
    'AAPL', 
    'UBER', 
    'ADP', 
    'META', 
    'AMD',
    'GOOG',
    'AMC'
]
company_to_ticker = {
    'NVIDIA CORP': 'NVDA',
    'APPLE INC': 'AAPL',
    'UBER': 'UBER', 
    'AUTOMATIC DATA PROCESSING INC': 'ADP',
    'META': 'META',
    'AMD': 'AMD',
    'ALPHABET': 'GOOG',
    'ALPHABET ': 'GOOG',
    'AMC': 'AMC'
}
keyword_db_path = "combined.db"
keyword_db_table = "analyzedDB"
max_depth_setting = 5
max_num_clusters = 5

nlp = spacy.load("en_core_web_sm")

## Read in stock percentage stock price data
# Define the end date as December 1, 2023
end_date = datetime.date(2023, 12, 1)

# Number of years you want to calculate the percentage change for
num_years = 25

# Initialize an empty list to store the percentage changes
price_changes = []

# Loop through each year and calculate the percentage change for each stock
for ticker in tickers:
    for year in range(end_date.year, end_date.year - num_years, -1):
        start_date = datetime.date(year, 1, 1)
        end_date_year = datetime.date(year, 12, 31)
        try:
            # Fetch historical data for the stock
            stock_data = yf.download(ticker, start=start_date, end=end_date_year)

            # Check if the data is empty
            if stock_data.empty:
                print(f"Warning: No data found for {ticker} in {year}.")
                percentage_change = np.nan  # Store NaN for missing data
            else:
                # Get the closing price at the start and end of the year
                start_price = stock_data['Close'].iloc[0]  # First day of the year
                end_price = stock_data['Close'].iloc[-1]   # Last day of the year

                # Calculate the percentage change
                percentage_change = ((end_price - start_price) / start_price) * 100

        except Exception as e:
            print(f"Error fetching data for {ticker} in {year}: {e}")
            percentage_change = np.nan  # Store NaN for failed requests

        # Append the result (ticker, year, percentage change)
        price_changes.append((ticker, year, percentage_change))

# Convert the list of tuples into a DataFrame
price_df = pd.DataFrame(price_changes, columns=['ticker', 'year', 'percentage_change'])

## Read in keyword data for stocks
## Connect to the source DB
conn = sqlite3.connect(keyword_db_path)
# Query from the keyword database the entries about business strategy
query = f"""SELECT * FROM {keyword_db_table} WHERE tag IN ('"Business"', '"RF"', '"URFC"', '"Cybersecurity"', '"Properties"', '"MDAFC"', '"QQDMR"', '"FSSD"') AND value IS NOT NULL;
"""
# Construct first merged dataframe from the query
kw_df_encoded = pd.read_sql(query, conn)
kw_df = kw_df_encoded.applymap(json.loads)
# Disconnect from the db
conn.close()
# Convert from company name to ticker
kw_df['ticker'] = kw_df['name'].map(company_to_ticker)
# Get year column
kw_df['year'] = kw_df['time'].astype(int)

## Join kw_df and price_df
merge_df1 = pd.merge(price_df, kw_df, on=['ticker', 'year'], how='inner')

## Construct joined df from all three tables
# df = pd.merge(merge_df1, sentiment_df, on=['ticker', 'year'], how='inner')
df = merge_df1
## NOTE: Uncomment this when we need Harrison to generate a new csv for the sentiments
# df.to_csv('out.csv')
# path_to_sentiment_added_csv = './sentimentadded.csv'
# df = pd.read_csv(path_to_sentiment_added_csv)
# print(df.head)

# print(df['extracted_summary_keywords'])
# df['extracted_summary_keywords'] = df['extracted_summary_keywords'].str.replace(',', ' ')
# df['extracted_summary_keywords'] = df['extracted_summary_keywords'].str.split()
# print(df['extracted_summary_keywords'])

# all_keywords = [keyword for sublist in df['extracted_summary_keywords'] for keyword in sublist]
all_keywords = []
for sublist in df['extracted_summary_keywords']:
    # Really unclean way to handle input coming from Harrison's csv because for some reason, the strings are not read as strings
    # sublist = sublist.replace('[', '')
    # sublist = sublist.replace(']', '')
    # sublist = sublist.replace(',', '')
    # sublist = sublist.replace("'", '')
    # all_keywords.extend(sublist.split())
    all_keywords.extend(sublist)
# Construct Boolean columns for each keyword
# for keyword in all_keywords:
    # df[keyword] = df['extracted_summary_keywords'].apply(lambda keywords: keyword in keywords if isinstance(keywords, list) else False)
# Convert the target variable ('stock_movement') into a categorical numerical format

accuracy_df = pd.DataFrame({'testing_ticker': [], 'num_clusters': [], 'max_depth': [], 'accuracy': []})

# Try building the model for several different numbers of hyperparameters
for num_clusters in range(2, max_num_clusters + 1):
    for max_depth_setting in range(2, max_depth_setting + 1):
        for ticker in df['ticker'].unique():
            with open(f'hidden-stock-output/depth-{max_depth_setting}-num_clusters-{num_clusters}-{ticker}.txt', 'w') as f: 
                # Construct most important keywords by k-means clustering algorithms
                kw_embeddings = []
                for kw in all_keywords:
                    word_vector = nlp(kw).vector
                    kw_embeddings.append(word_vector)
                kw_embeddings = np.array(kw_embeddings)
                kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
                kmeans.fit(kw_embeddings)
                cluster_labels = kmeans.labels_
                kw_to_cluster = {keyword: cluster for keyword, cluster in zip(all_keywords, cluster_labels)}

                # Prepare to recover representative words from cluster and output cluster membership
                # Not used in the computation, but is useful for the interpretability of the results from the decision tree
                centroids = kmeans.cluster_centers_
                def get_cosine_similarity(vec1, vec2):
                    return cosine_similarity([vec1], [vec2])[0][0]
                cluster_representatives = {}
                for cluster_num in range(num_clusters):
                    cluster_keywords = [keyword for keyword, label in zip(all_keywords, cluster_labels) if label == cluster_num]
                    cluster_vectors = [kw_embeddings[i] for i, label in enumerate(cluster_labels) if label == cluster_num]
                    centroid = centroids[cluster_num]
                    word_similarities = []
                    for i, word in enumerate(cluster_keywords):
                        similarity = get_cosine_similarity(cluster_vectors[i], centroid)
                        word_similarities.append((word, similarity))
                    sorted_similarities = sorted(list(set(word_similarities)), key=lambda x: x[1], reverse=True)
                    # print(sorted_similarities)
                    representative_words = [word for word, _ in sorted_similarities[:3]]
                    cluster_representatives[cluster_num] = representative_words
                for cluster_num, words in cluster_representatives.items():
                    print(f"Cluster {cluster_num} representatives: {', '.join(words)}", file=f)    
                for i in range(num_clusters):
                    print(f'Cluster number {i}:', file=f)
                    print([k for k,v in kw_to_cluster.items() if v == i], file=f)

                # create features for each of the clusters
                def count_cluster_keywords(keywords_list, keyword_to_cluster, num_clusters):
                    # Create a list to hold the count of words in each cluster
                    cluster_counts = np.zeros(num_clusters, dtype=int)
                    
                    # For each keyword in the list, increment the corresponding cluster's count
                    for keyword in keywords_list:
                        if keyword in keyword_to_cluster:
                            cluster_id = keyword_to_cluster[keyword]
                            cluster_counts[cluster_id] += 1
                    
                    return pd.Series(cluster_counts, index=[f'cluster_{i}' for i in range(num_clusters)])

                cluster_columns = [f'cluster_{i}_focus' for i in range(num_clusters)]
                df[cluster_columns] = df['extracted_summary_keywords'].apply(lambda x: count_cluster_keywords(x, kw_to_cluster, num_clusters))

                # Maybe try to use DBSCAN later?
                df['stock_movement'] = pd.cut(df['percentage_change'], bins=[-float('inf'), -3, 3, float('inf')], labels=[-1, 0, 1])
                # Construct lagged stock_movement columns?
                # print(df['stock_movement'])

                # Define the feature columns
                features = []
                features = list(cluster_columns)
                # features.append('sentiment')

                train_data = df[df['ticker'] != ticker]
                test_data = df[df['ticker'] == ticker]
                
                X_train = train_data[features]
                y_train = train_data['stock_movement']
                X_test = test_data[features]
                y_test = test_data['stock_movement']

                ## Actually building the decision tree

                ## Build Decision Tree Classifier
                # Initialize the decision tree classifier
                clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth_setting)

                # Train the model on the training data
                clf.fit(X_train, y_train)

                ## Evaluate the Model
                # Predict on the test set
                y_pred = clf.predict(X_test)

                # Evaluate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {accuracy * 100:.2f}%", file=f)

                # Print classification report (Precision, Recall, F1-score)
                print("Classification Report:", file=f)
                print(classification_report(y_test, y_pred, zero_division = 1), file=f)

                # Print confusion matrix
                print("Confusion Matrix:", file=f)
                print(confusion_matrix(y_test, y_pred), file=f)

                ## Visualize the Decision Tree (Optional)
                from sklearn.tree import plot_tree
                import matplotlib.pyplot as plt

                # Visualize the Decision Tree with detailed information (split rules)
                plt.figure(figsize=(30, 30))
                plot_tree(clf, 
                        feature_names=features,  # Names of the features used in splitting
                        class_names=['decrease', 'little to no change', 'increase'],  # Target labels
                        filled=True,  # Color nodes based on the predicted class
                        rounded=True,  # Make node edges rounded
                        fontsize=10,  # Set font size for clarity
                        precision=2)  # Precision of values displayed (e.g., thresholds)
                plt.show()
                plt.savefig(f'hidden-stock-output/depth-{max_depth_setting}-num_clusters-{num_clusters}-{ticker}.png')
                plt.close()
                
                new_row = pd.DataFrame({'testing_ticker': [ticker], 'num_clusters': [num_clusters], 'max_depth': [max_depth_setting],  'accuracy': [accuracy]})
                accuracy_df = pd.concat([accuracy_df, new_row], ignore_index=True)

accuracy_df.to_csv('hidden-stock-output/accuracy_data.csv')