
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
import numpy as np
import plotly.express as px
#% matplotlib inline
plt.style.use('Solarize_Light2')

#r = requests.get('https://datamarket.com/api/v1/list.json?ds=22qx')
#jobj = json.loads(r.text[18:-1])
#data = jobj[0]['data']
#df = pd.DataFrame(data, columns=['time', 'data']).set_index('time')

def plot(data: pd.DataFrame):
    fig = px.line(data, x=data.index, y=data['Volume'], title='Yahoo Stock')
    fig.show()
    return

df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/yahoo_stock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', drop=True, inplace=True)
plot(df)
df = df[['Volume']]
df.columns = ['data']
print('ts data')
print(df.head())
train = df.iloc[100:-10, :]
test = df.iloc[-10:, :]
train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)
pred = test.copy()
print(train.index)
print(test.index)

def expSmooth(dfUsage: DataFrame)-> Series:
    ses = SimpleExpSmoothing(dfUsage)
    alpha = 0.2
    model = ses.fit(smoothing_level=alpha, optimized=False)
    forecastperiods = 3
    forecast = model.forecast(forecastperiods)
    print('Exponential Smoothing')
    print(forecast)
    return forecastze

model = SimpleExpSmoothing(np.asarray(train['data']))
model._index = pd.to_datetime(train.index)

fit1 = model.fit()
pred1 = fit1.forecast(9)
fit2 = model.fit(smoothing_level=.2)
pred2 = fit2.forecast(9)
fit3 = model.fit(smoothing_level=.5)
pred3 = fit3.forecast(9)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[150:], train.values[150:])
ax.plot(test.index, test.values, color="gray")
for p, f, c in zip((pred1, pred2, pred3), (fit1, fit2, fit3), ('#ff7823', '#3c763d', 'c')):
    ax.plot(train.index[150:], f.fittedvalues[150:], color=c)
    ax.plot(test.index, p, label="alpha=" + str(f.params['smoothing_level'])[:3], color=c)
plt.title("Simple Exponential Smoothing")
plt.legend();

model = Holt(np.asarray(train['data']))
model._index = pd.to_datetime(train.index)

fit1 = model.fit(smoothing_level=.3, smoothing_slope=.05)
pred1 = fit1.forecast(9)
fit2 = model.fit(optimized=True)
pred2 = fit2.forecast(9)
fit3 = model.fit(smoothing_level=.3, smoothing_slope=.2)
pred3 = fit3.forecast(9)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train.index[150:], train.values[150:])
ax.plot(test.index, test.values, color="gray")
for p, f, c in zip((pred1, pred2, pred3), (fit1, fit2, fit3), ('#ff7823', '#3c763d', 'c')):
    ax.plot(train.index[150:], f.fittedvalues[150:], color=c)
    ax.plot(test.index, p,
            label="alpha=" + str(f.params['smoothing_level'])[:4] + ", beta=" + str(f.params['smoothing_slope'])[:4],
            color=c)
plt.title("Holt's Exponential Smoothing")
plt.legend();