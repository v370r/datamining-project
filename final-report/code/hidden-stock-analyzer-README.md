# hidden-stock-analyzer.py

## Overview

This script constructs graphs for the statistics collected by 
`decision-tree-hidden-stock.py`. You will need to run that before you run this 
so the statistics are generated.

## Expected Inputs

This script requires a file called `accuracy_data.csv` in a directory called 
`hidden-stock-output` in the same directory as the script 
`hidden-stock-analyzer.py`. This file can be generated by using 
`decision-tree-hidden-stock.py`.

## Expected Outputs
There are two types of outputs. These outputs will be left in the 
`hidden-stock-output` directory.
- `accuracy_plot_{company_ticker}.png `: These are plots with hyperparameter 
settings as independent variables and accuracy of the decision tree as 
dependent variables.
- `average_accuracy_plot.png`: This is a plot with the hidden stock as the 
independent variable and the average accuracy over all hyperparameter 
settings as the dependent variable.

## Usage
You will need to run `decision-tree-hidden-stock.py`. 

After you have run that command, you can run this script by running the 
following command:
```sh
python3 hidden-stock-analyzer.py
```