# decision-tree-hidden-stock README

## Overview

This script constructs decision trees for classifying whether a stock price is 
expected to increase, stay the same, or decrease, based on the keywords of the 
company associated to the stock. The script constructs several decision trees 
by tuning a few hyperparameters (see the included report for more details). 
Statistics are collected for each of the decision trees by hiding one stock, 
training the tree on the rest of stocks, and then measuring the accuracy of 
the tree on the hidden stock. Several decision tree images and statistics 
files are output.

## Expected Inputs

This script (by default) requires that a SQLite database file named 
`combined.db` with a table named `analyzedDB` is contained. The table 
`analyzedDB` should have the following columns (not all the columns are used):
- tag: This indicates what section of the document an excerpt is from.
- type: This indicates the type of document an excerpt is from.
- value: This is document excerpt.
- time: This is the time period the document refers to.
- name: This is the name of the company releasing the document.
- cleaned_text: The text with capitalization and punctuation removed.
- tokens: The cleaned text separated into single words.
- cleaned_text_topic: A number indicating the topic of the document excerpt.
- raw_text_keywords: Keywords extracted from `value`.
- cleaned_summary: Extractive summary of `value` with capitalization and 
punctuation removed.
- extracted_summary_keywords: Keywords extracted from `cleaned_summary`.

A database of this form can be generated using `keyword-topicmodelling.py` if 
you do not have a database of this form. 

There are several parameters that you can adjust in the code:
- tickers: The set of companies to download stock data for
- company_to_ticker: A dictionary to convert from stock names to ticker symbols
- keyword_db_path: The path to the database mentioned above.
- keyword_db_table: The name of the table above.
- max_depth_setting: The maximum depth for the decision trees generated.
- max_num_clusters: The maximum number of clusters to combine the keywords 
into. 

## Expected Outputs
There are three types of outputs.
- `accuracy_data.csv`: This summarizes the accuracy with columns indicating 
the settings of hyperparameters, which stock was hidden, and the final 
accuracy.
- `depth-{max_depth}-num_clusters-{num_clusters}-{hidden_stock}.png`: The 
decision tree generated by setting `max_depth`, `num_clusters`, and 
`hidden-stock` appropriately.
- `depth-{max_depth}-num_clusters-{num_clusters}-{hidden_stock}.csv`: The 
clusters and interpretations generated by the algorithm, and the accuracy 
statistics computed for the decision tree generated by setting `max_depth`, 
`num_clusters`, and `hidden-stock` appropriately.

## Usage
You will need to set the database path correctly prior to using this program.

You will need to create folders to hold the output for this program. This 
script expects a folder with the name `hidden-stock-output` in the same 
directory as the script. To ensure this, navigate to the directory containing 
this script, and run the following command:
```
mkdir hidden-stock-output
```
After you have created the folder, you can run this script by running the 
following command:
```
python3 decision-tree-hidden-stock.py
```