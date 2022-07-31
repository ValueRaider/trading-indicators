import yfinance as yf

dxy_df = yf.download("DX-Y.NYB", period="max")
dxy_df.to_csv("dxy.csv")
