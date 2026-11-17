# Order Flow Imbalance Feature Engineering

Order flow imbalance (OFI) features built from Level 10 limit order book data for AAPL, and a test of whether they predict short-term price moves. The features follow Cont, Cucuringu and Zhang, [*Cross-impact of order flow imbalance in equity markets*](https://arxiv.org/abs/2112.13213).

## Data

`first_25000_rows.csv`: 25,000 order book updates from October 21, 2024 (about 70 minutes). Each row has the top 10 bid and ask levels with prices, sizes and order counts.

## Features (`main.py`)

All features are summed over 5-second intervals.

- **Best-level OFI:** the change in buying versus selling pressure at the best bid and ask.
- **Multi-level OFI:** the same calculation at each of the 10 levels, divided by the average order book depth so the levels are comparable.
- **Integrated OFI:** the 10 levels combined into one number, weighted by the first principal component (PCA).
- **Cross-asset OFI:** code to line up OFI from several stocks by timestamp. It currently runs on AAPL data only.

Outputs are saved to `feature_outputs/`.

## Analysis (`regressionexploration.ipynb`)

1. **Targets:** price changes 5, 10, 15, 30, 45 and 60 seconds ahead.
2. **Stationarity:** ADF tests confirm every feature and target is stationary.
3. **Lag selection:** autocorrelation plots (95% confidence) show which lags of each feature carry information.
4. **Models:** Ridge regression, random forest and ARIMAX on the lagged features, predicting the 5-second price change.

## Results

- In the ARIMAX model, two lag structures were statistically significant at the 5% level:
  - **60 seconds:** level 0 and level 4 OFI, both at lag 12 (p = 0.003 and 0.006)
  - **3–4 minutes:** level 4 OFI at lag 34 and level 9 OFI at lag 44 (p = 0.010 and 0.016)
- The PCA-integrated OFI was not significant once lagged. Almost all of its weight landed on a single level, so it lost the information in the others.
- Out-of-sample predictive power was close to zero: Ridge test R² was 0.002, and the random forest overfit (train R² 0.28, test −0.05). With only 70 minutes of data, these results are a starting point for 1-minute prediction rather than a usable signal.

## Running it

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib
python main.py                      # writes the feature CSVs
jupyter notebook regressionexploration.ipynb
```
