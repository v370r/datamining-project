# Multi-Stock Price Prediction Using Sentiment Analysis

This repository contains a machine learning project to predict stock prices using historical data and sentiment analysis. The pipeline includes data preprocessing, dataset construction, model training with LSTM, and evaluation.

## Features

- **Data Collection**: Utilizes `yfinance` to gather historical stock price data and a CSV file containing sentiment analysis for financial reports.
- **Sentiment Mapping**: Maps sentiment tags to numerical values for feature inclusion.
- **Custom Dataset**: Implements a PyTorch `Dataset` class to prepare data for training and evaluation.
- **LSTM-based Model**: Predicts future stock prices using both time-series and sentiment data.
- **Visualization**: Plots predictions against historical data for analysis.

---

## Requirements

- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - torch
  - yfinance
  - matplotlib

Install the required libraries using pip:
```bash
pip install numpy pandas torch yfinance matplotlib
```

---

## Data Preparation

1. **Input Files**:
   - `sentimentadded.csv`: Contains columns such as `time`, `ticker`, `tag`, `sentiment`, and `percentage_change`.
   - Historical stock data is fetched automatically using the `yfinance` library.

2. **Preprocessing**:
   - Sentiment tags (`tag`) are mapped to a fixed index.
   - Stock price data is normalized using z-scores based on the standard deviation.
   - Time-series data is split into windows of 60 days.

---

## Code Overview

### Key Components

#### 1. **Data Loading**
```python
df = pd.read_csv("sentimentadded.csv")
stock_data = {}
for ticker in df["ticker"].unique():
    t = yf.Ticker(ticker)
    stock_data[ticker] = t.history('max')
```

#### 2. **Dataset Class**
The `StockDataset` class handles the creation of input features and labels.
- Sentiment data is converted to a numerical vector.
- Stock price data is split into overlapping windows.
- Each sample consists of a window of stock prices, a sentiment vector, and a corresponding label.

#### 3. **Model Definition**
The `PricePredictor` model combines:
- An LSTM layer to process time-series data.
- Fully connected layers to incorporate sentiment data.

```python
class PricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stock_layer = nn.LSTM(1, 512, 5)
        self.sent_layer = nn.Sequential(
            nn.Linear((512 * 60) + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )

    def forward(self, stock, sent):
        lstm_out = self.stock_layer(stock)[0].flatten(start_dim=1)
        return self.sent_layer(torch.hstack((lstm_out, sent)))
```

#### 4. **Training and Testing**
The training loop uses the Mean Squared Error loss and the Adam optimizer.
```python
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for stockx, sentx, y in dataloader:
        stockx, sentx, y = stockx.to(device), sentx.to(device), y.to(device)
        pred = model(stockx.unsqueeze(2), sentx)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multi-stock-predictor.git
cd multi-stock-predictor
```

2. Run the script:
```bash
python main.py
```

---

## Results

- The model predicts normalized stock price deviations based on both time-series data and sentiment features.
- Example prediction for AAPL (Apple Inc.):

![AAPL Prediction Plot](plot.png)

---

## Future Work

- Expand sentiment analysis with natural language processing (NLP) for textual data.
- Test with more complex models such as Transformers.
- Incorporate additional features like trading volume and macroeconomic indicators.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Yahoo Finance](https://finance.yahoo.com/) for data access.
- Contributors to open-source libraries used in this project.

