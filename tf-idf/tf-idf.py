import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):  # assuming text files
            with open(file_path, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords and non-alphanumeric tokens
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)

# We need to use the custom tokenizer if you want to look for specific patterns in words
def custom_tokenizer(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text.lower())
    # Create bigrams (pairs of consecutive words)
    bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words)-1)]
    return bigrams

folder_path = './input'  # Replace with your folder path
documents = load_texts_from_folder(folder_path)
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Apply TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,3))
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Get feature names (words) corresponding to the TF-IDF values
feature_names = vectorizer.get_feature_names_out()

# Sum the TF-IDF scores for each word across all documents
tfidf_scores = tfidf_matrix.sum(axis=0).A1

# Create a dictionary of words and their corresponding scores
word_scores = dict(zip(feature_names, tfidf_scores))

# Sort the words by their TF-IDF scores in descending order
sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

# Show top 10 keywords
for word, score in sorted_keywords:
    print(f'{word}: {score}')
