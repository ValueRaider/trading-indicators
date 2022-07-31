import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
fix_size_x = 24
fix_size_y = 12

dxy_df = pd.read_csv("dxy.csv")
dxy_df["datetime"] = pd.to_datetime(dxy_df["Date"])
dxy_df.drop("Date", axis=1, inplace=True)
dxy_df.drop("Open", axis=1, inplace=True)
dxy_df.drop("Low", axis=1, inplace=True)
dxy_df.drop("High", axis=1, inplace=True)
dxy_df.drop("Adj Close", axis=1, inplace=True)
dxy_df.drop("Volume", axis=1, inplace=True)

from indicators import *

sma20 = SMA(20)
dxy_df = sma20.AddToFrame(dxy_df, "Close")

sd20 = StdDev(20)
dxy_df = sd20.AddToFrame(dxy_df, "Close")

bb_upper = HybridIndicator(1.0, sma20, 2.0, sd20)
dxy_df = bb_upper.AddToFrame(dxy_df, "Close")
bb_lower = HybridIndicator(1.0, sma20, -2.0, sd20)
dxy_df = bb_lower.AddToFrame(dxy_df, "Close")

# start_d = pd.Timestamp("2020-01-01")
start_d = pd.Timestamp("2020-08-20")
#start_d = pd.Timestamp("2021-01-01")
dxy_df = dxy_df[dxy_df["datetime"] > start_d].reset_index()


def CalcSmaCrossover(df, sma1_period, sma2_period):
	p1 = sma1_period
	p2 = sma2_period
	str1 = "sma{0}".format(p1)
	str2 = "sma{0}".format(p2)
	df[str1] = df["Close"].rolling(window=p1, center=False).mean()
	df[str2] = df["Close"].rolling(window=p2, center=False).mean()

	str_dlt = "delta_{0}_{1}".format(str1, str2)
	df[str_dlt] = df[str1] - df[str2]

	str_shft = "delta_{0}_{1}_sh".format(str1, str2)
	df[str_shft] = df[str_dlt].shift(1)

	str_crU = "crossUp_{0}_{1}".format(str1, str2)
	df[str_crU] = np.logical_or(np.logical_and(df[str_dlt] > 0.0, df[str_shft] <= 0.0), 
													np.logical_and(df[str_dlt] >= 0.0, df[str_shft] < 0.0))

	str_crD = "crossDown_{0}_{1}".format(str1, str2)
	df[str_crD] = np.logical_or(np.logical_and(df[str_dlt] < 0.0, df[str_shft] >= 0.0), 
													np.logical_and(df[str_dlt] <= 0.0, df[str_shft] > 0.0))

	return df

# p1 = 25
# p2 = 200
# dxy_df2 = CalcSmaCrossover(dxy_df, p1, p2)
# f = dxy_df2["crossUp_sma25_sma200"]==True
# print(dxy_df2[f])
# quit()


# Plot:
fig = figure(num=None, figsize=(fix_size_x, fix_size_y), dpi=80, facecolor='w', edgecolor='k')
plt.plot(dxy_df["datetime"], dxy_df["Close"], label="Close")
plt.plot(dxy_df["datetime"], dxy_df["sma20"], label="sma20")

tolerance = 0.00075

bbu = bb_upper.Name()
plt.plot(dxy_df["datetime"], dxy_df[bbu], label=bbu)
# ratio = 1-tolerance
# plt.plot(dxy_df["datetime"], dxy_df[bbu]*ratio, label=bbu+"_{0}%".format(ratio*100.0))

bbl = bb_lower.Name()
plt.plot(dxy_df["datetime"], dxy_df[bbl], label=bbl)
# ratio = 1+tolerance
# plt.plot(dxy_df["datetime"], dxy_df[bbl]*ratio, label=bbl+"_{0}%".format(ratio*100.0))

# ratio = 1-tolerance
# plt.plot(dxy_df["datetime"], dxy_df["sma20"]*ratio, label="sma20_{0}%".format(ratio*100.0))
# ratio = 1+tolerance
# plt.plot(dxy_df["datetime"], dxy_df["sma20"]*ratio, label="sma20_{0}%".format(ratio*100.0))

plt.legend()
png_filename = "dxy-sma.png"
plt.savefig(png_filename)
plt.close()
