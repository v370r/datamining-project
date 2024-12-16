# direction-analysis-spacy.py

## Overview

This script computes a k-means clustering of companies by using the embeddings 
of the keywords associated to the company. This script tries several values of 
k and records the clusters which are best according to silhouette score. It also 
computes inertia values for clustering and records clusterings and 
interpretations for the clusterings. 

## Expected Inputs

This script requires a file called `company_keywords.csv` in the same 
directory as the script. You can control the number of clusters used by the 
algorithm by adjusting the variable `best_k` in the script.

## Expected Outputs

There are two types of outputs. 
- `output-cluster-scores-spacy.png`: This graphs both the inertia and the 
silhouette scores of the k-means clustering as a function of the number of 
clusters to generate.
- `output-clustering-spacy.png`: This is the clustering output by the setting 
of `best_k`.

## Usage

You can run this script by running the 
following command:
```
python3 direction-analysis-spacy.py
```
Some output is printed to standard output, so to collect this into a file, you 
will need to pipe the output to a file like so:
```
python3 direction-analysis-spacy.py > clusters.txt
```