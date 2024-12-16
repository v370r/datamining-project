# keyword-topicmodelling.py

## Overview

This Python script provides a pipeline for performing various text processing 
and analysis tasks on a document corpus. It leverages multiple natural 
language processing (NLP) techniques such as preprocessing, keyword 
extraction, topic modeling, and summarization. The script is designed to 
process text data from a SQLite database, generate text summaries, and store 
the results back into a database.

## Expected Inputs

This script (by default) requires that a SQLite database file named 
`combined.db` with a table named `processdDB` is contained. The table 
`processdDB` should have the following columns (not all the columns are used):
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

## Expected Outputs

This script outputs a table `analyzedDB`. The table `analyzedDB` has the 
following columns:
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

## Usage

You can run this script by running the 
following command:
```
python3 keyword-topicmodelling.py
```