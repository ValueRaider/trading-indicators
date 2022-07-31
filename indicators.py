import numpy as np
import pandas as pd

import enum
from pprint import pprint

class IndicatorType(enum.Enum):
	Price     = 1
	StdDev    = 2
	MoneyFlow = 3
	SMA       = 11
	HMA       = 12

class Pattern:
	## A series of events. An event is an interaction between 
	## price and an Indicator, or between two Indicators. Can be: 
	## - crossover
	## - bounce

	## Thus pattern involves one or several Indicators.
	## Pattern specifies interval between events - will need to be loose, e.g. 
	## E2 occurs 5-15 days after E1

	def __init__(self):
		self.sequence = []

class EventType(enum.Enum):
	Bounce = 1
	CrossUp = 2
	CrossDown = 3

class Event:
	def __init__(self, etype):
		if not isinstance(etype, EventType):
			raise TypeError("class Event init() failed: arg1 expected EventType but passed {0}".format(type(etype)))
		self.etype = etype

class Bounce:
	## Each bounce event has its own tolerance (how close to price/indicator to be counted as a bounce)
	## Store specific series name / indicator?
	def __init__(self, tolerance):
		if not isinstance(tolerance, float):
			raise TypeError("class Bounce init() failed: arg1 expected float but passed {0}".format(type(tolerance)))
		super().__init__(EventType.Bounce)
		self.tolerance = tolerance

class Indicator:
	def __init__(self, itype):
		if not isinstance(itype, IndicatorType):
			raise TypeError("class Indicator init() failed: arg1 expected IndicatorType but passed {0}".format(type(itype)))
		self.itype = itype

	def Calculate(self, series=None):
		raise NotImplementedError("class Indicator Calculate() failed: is virtual method")

	def AddToFrame(self, frame=None):
		raise NotImplementedError("class Indicator AddToFrame() failed: is virtual method")

class SMA(Indicator):
	def __init__(self, window):
		if not isinstance(window, int):
			raise TypeError("class SMA init() failed: arg1 expected int but passed {0}".format(type(window)))
		super().__init__(IndicatorType.SMA)
		self.window = window

	def Name(self):
		return "sma{0}".format(self.window)
		# return "{0}dMA".format(self.window)

	def Calculate(self, series):
		if not isinstance(series, pd.Series):
			raise TypeError("class SMA Calculate() failed: arg1 expected pd.Series but passed {0}".format(type(series)))
		return series.rolling(window=self.window, center=False).mean()

	def AddToFrame(self, frame, seriesName):
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("class SMA AddToFrame() failed: arg1 expected pd.DataFrame but passed {0}".format(type(frame)))
		if not isinstance(seriesName, str):
			raise TypeError("class SMA AddToFrame() failed: arg2 expected str but passed {0}".format(type(seriesName)))

		if not seriesName in frame.columns:
			raise Error("No column named '{0}' in frame".format(seriesName))

		colName = self.Name()
		if not colName in frame.columns:
			frame[colName] = self.Calculate(frame[seriesName])
		return frame

class StdDev(Indicator):
	def __init__(self, window):
		if not isinstance(window, int):
			raise TypeError("class StdDevs init() failed: arg1 expected int but passed {0}".format(type(window)))
		super().__init__(IndicatorType.StdDev)
		self.window = window

	def Name(self):
		return "sm-std{0}".format(self.window)

	def Calculate(self, series):
		if not (isinstance(series, pd.Series) or isinstance(series, pd.DataFrame)):
			raise TypeError("class StdDev Calculate() failed: arg1 expected pd.Series or .DataFrame but passed {0}".format(type(series)))
		if isinstance(series, pd.DataFrame):
			## This is a little trick. Need to collapse into single series
			l = series.shape[0]
			w = series.shape[1]
			lw = l*w
			if w == 2:
				s0 = series[series.columns.values[0]]
				s1 = series[series.columns.values[1]]
				sz = [v for z in zip(s0, s1) for v in z]
				sz = pd.Series(sz)
				sz_std = sz.rolling(window=self.window*w, center=False).std()
				## Keep one value per window:
				indices = list(range(0, lw, w))
				sz_std = sz_std[indices].reset_index(drop=True)
				return sz_std
		return series.rolling(window=self.window, center=False).std()

	def AddToFrame(self, frame, seriesName):
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("class StdDev AddToFrame() failed: arg1 expected pd.DataFrame but passed {0}".format(type(frame)))
		if isinstance(seriesName, list):
			for sn in seriesName:
				if not isinstance(sn, str):
					raise TypeError("class StdDev AddToFrame() failed: arg2 expected (list of) str but passed list with {0}".format(type(sn)))
		elif not isinstance(seriesName, str):
			raise TypeError("class StdDev AddToFrame() failed: arg2 expected str but passed {0}".format(type(seriesName)))

		if isinstance(seriesName, list):
			for sn in seriesName:
				if not sn in frame.columns:
					raise Error("No column named '{0}' in frame".format(sn))
		elif not seriesName in frame.columns:
			raise Error("No column named '{0}' in frame".format(seriesName))

		colName = self.Name()
		if not colName in frame.columns:
			frame[colName] = self.Calculate(frame[seriesName])
		return frame

class MoneyFlow(Indicator):
	def __init__(self, window):
		if not isinstance(window, int):
			raise TypeError("class MoneyFlow init() failed: arg1 expected int but passed {0}".format(type(window)))
		super().__init__(IndicatorType.MoneyFlow)
		self.window = window

	def Name(self):
		return "moneyflow{0}".format(self.window)

	def Calculate(self, df):
		if not isinstance(df, pd.DataFrame):
			raise TypeError("class MoneyFlow Calculate() failed: arg1 expected pd.DataFrame but passed {0}".format(type(df)))

		expected_cols = ["High", "Low", "Close", "Volume"]
		missing_cols = [c for c in expected_cols if not c in df.columns.values]
		if len(missing_cols) > 0:
			raise Exception("class MoneyFlow Calculate() failed: arg1 is missing these columns: {0}".format(missing_cols))

		## Using formulas from www.investopedia.com/terms/m/mfi.asp
		d = df.copy()
		n = self.window+1
		if n > d.shape[0]:
			raise Exception("class MoneyFlow Calculate() failed: arg1 needs at least {0} rows, but it has {1}".format(n, d.shape[0]))

		d = d[["High", "Low", "Close", "Volume"]]
		d["Typical price"] = (d["High"]+d["Low"]+d["Close"])*0.33333

		d["Raw money flow"] = d["Typical price"]*d["Volume"]

		d2 = d.iloc[1:d.shape[0]].reset_index(drop=True)
		d2["Delta"] = np.subtract(d["Typical price"].values[1:d.shape[0]] , d["Typical price"].values[0:d.shape[0]-1])
		d = d2

		n = d.shape[0]
		deltas = d["Delta"].values
		flows = d["Raw money flow"].values
		mfi = np.zeros(n)
		for w_start in range(0, n-self.window+1):
			w_flows = flows[w_start:w_start+self.window]
			window_pos_f = np.where(deltas[w_start:w_start+self.window]>=0.0)
			window_neg_f = np.where(deltas[w_start:w_start+self.window] <0.0)
			pos_money_flow = np.sum(w_flows[window_pos_f])
			neg_money_flow = np.sum(w_flows[window_neg_f])
			if neg_money_flow == 0.0:
				money_flow_index = 0.0
			else:
				money_flow_ratio = pos_money_flow / neg_money_flow
				money_flow_index = 100 - 100/(1+money_flow_ratio)
			mfi[w_start+self.window-1] = money_flow_index
		return mfi

	def AddToFrame(self, frame):
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("class MoneyFlow AddToFrame() failed: arg1 expected pd.DataFrame but passed {0}".format(type(frame)))

		colName = self.Name()
		if not colName in frame.columns:
			frame[colName] = self.Calculate(frame)
		return frame

class HybridIndicator(Indicator):
	def __init__(self, a1, i1, a2, i2):
		if not isinstance(a1, float):
			raise TypeError("class HybridIndicator init() failed: arg1 expected float but passed {0}".format(type(a1)))
		if not isinstance(i1, Indicator):
			raise TypeError("class HybridIndicator init() failed: arg2 expected Indicator but passed {0}".format(type(i1)))
		if not isinstance(a2, float):
			raise TypeError("class HybridIndicator init() failed: arg3 expected float but passed {0}".format(type(a2)))
		if not isinstance(i2, Indicator):
			raise TypeError("class HybridIndicator init() failed: arg4 expected Indicator but passed {0}".format(type(i2)))

		self.a1 = a1
		self.i1 = i1
		self.a2 = a2
		self.i2 = i2

	def Name(self):
		n = "{0}{1}".format(self.a1, self.i1.Name()) + "+" "{0}{1}".format(self.a2, self.i2.Name())
		return n

	def Calculate(self, series):
		if not isinstance(series, pd.Series):
			raise TypeError("class StdDev Calculate() failed: arg1 expected pd.Series but passed {0}".format(type(series)))

		i1d = self.i1.Calculate(series)
		i2d = self.i2.Calculate(series)
		return (i1d * self.a1) + (i2d * self.a2)

	def AddToFrame(self, frame, seriesName):
		if not isinstance(frame, pd.DataFrame):
			raise TypeError("class HybridIndicator AddToFrame() failed: arg1 expected pd.DataFrame but passed {0}".format(type(frame)))
		if not isinstance(seriesName, str):
			raise TypeError("class HybridIndicator AddToFrame() failed: arg2 expected str but passed {0}".format(type(seriesName)))

		if not seriesName in frame.columns:
			raise Error("No column named '{0}' in frame".format(seriesName))

		colName = self.Name()
		if not colName in frame.columns:
			frame[colName] = self.Calculate(frame[seriesName])
		return frame
