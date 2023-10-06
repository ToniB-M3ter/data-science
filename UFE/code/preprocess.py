from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
from tabulate import tabulate
import plot_data as plot

dfraw = pd.read_csv("/UFE/data/anon_evnt_dy_20220701_20230101.txt", sep='\t')
cols = ['timeStamp','org','customer','meterName','meterId','measureName','measureId','usage']
dfraw.columns = cols
dfraw.dropna(subset=['timeStamp'], inplace = True)
print('raw data length: ' + str(len(dfraw)))

syntheticMeters = ['balance','balance consumed','bill','commitment','commitment consumed','commitment fee',
'credit memo','debit memo','minimum spend','minimum spend refund','minimum spend/pricing band',
'overage surcharge','overage usage','overage usage/pricing band','service user','standing charge',
'standing charge/pricing band','usage','usage credit','usage credit/pricing band','usage/pricing band','user']

dfSynthetic = dfraw[dfraw['meterName'].isin(syntheticMeters)]
print('syntheticMeters data length: ' + str(len(dfSynthetic)))

dfUsage = dfraw[~dfraw['meterName'].isin(syntheticMeters)]
print('usageMeters data length: ' + str(len(dfUsage)))

#dfUsage.dropna(subset=['org'], inplace = True)  #Correct copy error
dfUsage = dfUsage.dropna(subset=['org'])
dfUsage['org'] = dfUsage['org'].astype(int)
dfUsage['timeStamp'] = pd.to_datetime(dfUsage['timeStamp'], format='%Y-%m-%d %H:%M:%S')
#dfUsage=dfUsage.set_index('timeStamp')
dfUsage.to_csv('dfUsage.csv')

# Display Group by to user in order to see which org/meter/measure have the highest counts
print('dfUsage groupby')
print(dfUsage.groupby(['org', 'timeStamp'], as_index = False).count().sort_values(['org', 'timeStamp'], ascending=True).head(20))
#print(dfUsage.groupby(['org', 'customer', 'customer','meterName','meterId','measureName','measureId'], as_index = False).nunique(['timeStamp']))
dfUsage['days_of_data'] = (dfUsage.groupby(['org', 'customer','meterName','meterId','measureName','measureId'])['timeStamp'].transform('nunique'))

dfUsageTop = dfUsage.sort_values(['timeStamp', 'days_of_data'], ascending=False).head(100)


#print('dfUsage nunique')
#print(tabulate(dfUsage.sort_values(['days_of_data','org', 'customer', 'timeStamp'], ascending= False).head(50), headers=cols, tablefmt='grid'))
