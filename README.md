# StockLens

## Overview

FIXME: 
This project is designed to process and aggregate data from various sources, including TSV files and SQLite databases. The primary goal is to extract, filter, and combine data related to specific companies, and then store the processed data in an SQLite database for further analysis.

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

TODO: add requirements.txt (pip freeze > requirements.txt)



## Usage
FIXME:
- `data-preprocessing/cleaning/`: Contains the main Jupyter notebooks for data aggregation, combination, and database connection.
  - `dataAggregator.ipynb`: Notebook for extracting and filtering data from TSV files.
  - `dataCombiner.ipynb`: Notebook for combining filtered data and appending/updating it in an SQLite database.
  - `connectSQLDB.ipynb`: Notebook for connecting to the SQLite database and performing SQL operations.
- `data-preprocessing/cleaning/Data/NVDA/processedData/`: Contains the processed data directories for different quarters.
  - Each quarter directory (e.g., `2019q4_notes/`, `2020q1_notes/`) contains `sub.tsv` and `txt.tsv` files.
- `extracted_summary_2024-12-03_145245.txt`: A text file containing extracted summaries from 10ks.
- `README.md`: The README file providing an overview and instructions for the project.

## Sequence
TODO:

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or suggestions, please contact vijay.poloju@colorado.edu, //TODO: add ur emails
