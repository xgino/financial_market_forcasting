# 01_data: get the data

Every player downloads and cleans data and writes CSV to `data/`.

- `yahoo_prices.py` daily prices for `settings.ASSETS`, in EUR, with the EUR/USD repair

Add a source: a new file with `SETTINGS` and a class based on `DataSource` with `run()`.
It returns a short report (shown on screen). Ideas: FRED (interest rates, dollar, VIX), Kaggle datasets.