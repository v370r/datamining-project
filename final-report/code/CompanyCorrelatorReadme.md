# Functions Overview

All functions mentioned below are located in the `companyCorrelator.ipynb` file. There is also a description of the database that this file interacts will below.

## Cross-Correlation Functions

### `correlateAndSaveAll(tech_tickers)`
This will compute the cross-correlation for all companies and save the results to the database. To use this function pass in:
- `tech_tickers` (List\<String\>): List of company tags.

### `crossCorrelation(tag1, tag2, displayLagGraph, printInfo, shouldSaveToDatabase, displayStockGraph)`
This computes the cross-correlation data between two companies. To use this function pass in:
- `tag1` (String): Company 1 tag
- `tag2` (String): Company 2 tag
- `displayLagGraph` (boolean): Whether to display a graph of computed lag values for both companies.
- `printInfo` (boolean): Whether to print additional info.
- `shouldSaveToDatabase` (boolean): Whether to save the results to the database.
- `displayStockGraph` (boolean): Whether to display an overlay of both stock graphs, accounting for lag.

## Volatility Functions

### `findVolatilityAndSaveAll(tech_tickers, includeFiveYear)`
This will find volatility for all the passed in tags and save it to the database. To use this function pass in:
- `tech_tickers` (List\<String\>): List of company tags.
- `includeFiveYear` (boolean): Determines if the program should include 5 year data as well as one year.

### `createVolatilityBarChart(tech_tickers, period, monthlyOrYearly)`
This will generate a bar chart for volatility. To use this function pass in:
- `tech_tickers` (List\<String\>): List of company tags.
- `period` (String): `"1y"` or `"5y"` (indicating the period of data to analyze).
- `monthlyOrYearly`(String): Can be `"monthly"`, `"yearly"`, or `"both"`. This determines the time frame for the data to be displayed (case-insensitive).

## Correlation Analysis Functions

All of these are executed on all the data stored in the table `CrossCorrelationResults` and column `max_correlation`
### `getCorrelationMean()`
Returns the mean of the correlation data.

### `getCorrelationStd()`
Returns the standard deviation of the correlation data.

### `getQuartiles()`
Returns a list containing:
- Index 0: First quartile
- Index 1: Median
- Index 2: Third quartile

## Grouping and Visualization

### `maxAVGCorrelationBetweenThreeCompAll(period, techTickers)`
To group companies into sets of three and find the average correlation. To use this function pass in:
- `period` (String): `"1y"` or `"5y"`
- `techTickers` (List\<String\>): Array of company tags.

This function returns a 2D array with:
- The first three columns: Company tags
- The fourth column: The average correlation value.
- Each Row is another group

### `GraphGroups(maxAVGCorrelationBetweenThreeCompAll, topAmountToGraph)`
This graphs the highest correlated groups. To use this function pass in:
- `maxAVGCorrelationBetweenThreeCompAll` (2d Array): The result from `maxAVGCorrelationBetweenThreeCompAll(period, techTickers)`
- `topAmountToGraph` (Int): The number of highest correlated groups to display in the graph.

## Database Structure

The database processdDB.db contains three tables:
- **Cross Correlation**: Stores two company tags, a `max_correlation` value, a `max_lag`, a `avg_correlation` value, a `period`, and a `description`.
- **Volatility**: Contains volatility data for each company. Stores `tag`, `monthlyVolatility`, `yearlyVolatility`, `period`, and `dateStored`
- **processdDB**: Contains sec data for each company. tag is not the company it is the type of data. name is the company. Stores `tag`, `type`, `value`, `time`, and `name`
