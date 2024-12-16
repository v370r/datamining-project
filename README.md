# StockLens

## Overview

This project involves sentiment analysis and findng meaningful relations between companies.

## Features

- Extract data from TSV files.
- Filter data based on specific criteria (e.g., company name, submission IDs).
- Combine filtered data into a single DataFrame.
- Append or update the combined data in an SQLite database.
- Perform SQL queries to retrieve and manipulate data.

## Project Structure

data-preprocessing/
│
├── cleaning/
│   ├── dataAggregator.ipynb
│   ├── dataCombiner.ipynb
│   └── Data/
│       └── NVDA/
│           └── processedData/                  This section is moved to DB
│               ├── 2019q4_notes/
│               │   ├── sub.tsv
│               │   └── txt.tsv
│               ├── 2020q1_notes/
│               │   ├── sub.tsv
│               │   └── txt.tsv
│               └── ...
├── extracted_summary_2024-12-03_145245.txt
├── requirements.txt
└── README.md

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/v370r/datamining-project
   
   cd datamining-project
   ```


2. **Install the required Python packages**

    ```sh
    pip install pandas sqlite3
    ``` 




## Usage
### Data Processing
  - `data-preprocessing/cleaning/`: Contains the main Jupyter notebooks for data aggregation, combination, and database connection.
    - `dataAggregator.ipynb`: Notebook for extracting and filtering data using multithreading from TSV files to processdb.
    - `dataCombiner.ipynb`: Notebook for combining filtered data and appending/updating  using multithreading it in an SQLite database processdb.
    - `connectSQLDB.ipynb`: Notebook for connecting to the SQLite database and performing SQL operations.
    - `dataDDownloader.pynb`: Download all the tsv zip files from edgar security and exchange filings using multithreading.
  - `data-preprocessing/cleaning/Data/NVDA/processedData/`: Contains the processed data directories for different quarters.
    - Each quarter directory (e.g., `2019q4_notes/`, `2020q1_notes/`) contains `sub.tsv` and `txt.tsv` files.
  - `extracted_summary_2024-12-03_145245.txt`: A text file containing extracted summaries from 10ks.
  - `README.md`: The README file providing an overview and instructions for the project.

### Company Correlator
  #### Overview

    This module calculates cross-correlation and volatility metrics for companies based on their stock data. The results are stored in a database and visualized for analysis. Detailed readme can be found by following below links 
    
  **Readme**: [CompanyCorrelatorReadme](./final-report/code/CompanyCorrelatorReadme.md)
  **CodePath**: [companyCorrelator.ipynb](./final-report/code/companyCorrelator.ipynb)

  #### Features
  - Cross-correlation computation for multiple companies.
  - Volatility analysis over 1-year or 5-year periods.
  - Grouping and visualization of top correlated company groups.

  #### Database Structure
  - **Cross Correlation**: Stores max correlation, lag, and average correlation values.
  - **Volatility**: Records monthly and yearly volatility for each company.

### Decision Tree Hidden Stock Analysis
  #### Overview
    This module uses decision trees to classify stock trends and measures accuracy by hiding one stock at a tim. THe results include decision tree and accuracy statistics. Detailed readme can be found by following below links 

  **Readme**: [decision-tree-hidden-stock-README](./final-report/code/decision-tree-hidden-stock-README.md)
  **CodePath**: [decision-tree](./final-report/code/decision-tree-hidden-stock.py)

### Direction Analysis Using SpaCy
  #### Overview
  This script clusters companies based on keyword embeddings, using k-means clustering to identify optimal groups based on silhouette scores and inertia.

  **Readme**: [direction-analysisi-readme](./final-report/code/direction-analysis-spacy-README.md)
  **CodePath**: [direction-analysis-code](./final-report/code/direction-analysis-spacy.py)

  #### Outputs
  - Cluster score graph [Click here](./final-report/Graph%20Pictures/output-cluster-scores-spacy.png)
  - Final Clustering visualization [Click here](./final-report/Graph%20Pictures/output-6-clustering-spacy.png)

### Hidden Stock Analyzer
  #### Overview
  Analyzes and visualizes statistics collected from the Decision Tree Hidden Stock script.

  **Readme**: [Hidden Stock Analyzer Readme](./final-report/code/hidden-stock-analyzer-README.md)
  **CodePath**: [Hidden Stock Analyzer](./final-report/code/hidden-stock-analyzer.py)

  #### Outputs
  - Accuracy plots for each stock
  - Average accuracy plot across all hyperparameter settings.

### Keyword Topic Modelling
  #### Overview
  Processes text data using NLP techniques, including keyword extraction, topic modeling, and summarization. Results are stored in the analyzedDB table in a SQLite database.

  **Readme**: [Keyword Topic Modelling](./final-report/code/keyword-topicmodelling-README.md)
  **CodePath**: [ Keyword Topic Modelling](./final-report/code/keyword-topicmodelling.py)

  #### Output
  - analyzedDB table with processed text and metadata.

### Multi-Stock Price Prediction Using Sentiment Analysis
  #### Overview
  Predicts stock prices using historical data and sentiment analysis with an LSTM-based model.

  **Readme**: [Multi-Stock Price Predictor](./final-report/code/sentiment_readme.md)
  **CodePath**: [Multi-Stock Price Prediction](./final-report/code/sentiment.py)

  #### Features
  - Custom PyTorch Dataset for handling stock price and sentiment data.
  - Model evaluation along with visualization of predictions.


### Sentiment Analysisi Using DistilBERT
#### Overview
A sentiment analysis tool using a fine-tuned DistilBERT model to process textual data and compute average sentiment scores.
#### Usage Example
```sh
from sentiment_analyzer import SentimentAnalyzer

sa = SentimentAnalyzer()
sa.add_text("Sample text here.")
sentiment_score = sa.get_sentiment_batch()
print(f"Sentiment Score: {sentiment_score}")
sa.reset()

```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or suggestions, please contact vijay.poloju@colorado.edu,
