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
print(dfUsage.groupby(['org', 'timeStamp']).count().sort_values(['usage'], ascending=False).head(50))

#create unique list of orgs
orgs = dfUsage['org'].unique() # 79 unique orgs

# create a data frame dictionary to store your data frames
OrgDict = {elem: pd.DataFrame() for elem in orgs}

for key in OrgDict.keys():
    OrgDict[key] = dfUsage[:][dfUsage.org == key]

def plot_data(df):
    plot.main(df)
    return

def return_org_df(org):
    org_df = OrgDict[org]
    org_df.sort_index(inplace=True)
    return org_df

def return_meter_df(org_df):
    # meters
    meters = org_df['meterName'].unique()
    meterDict = {elem: pd.DataFrame() for elem in meters}
    for key in meterDict.keys():
         meterDict[key] = org_df[:][org_df.meterName == key]

    # Display Group by of meters for selected org in order to choose meter/account
    print('organization grouped by meter')
    print(org_df.groupby(['meterName']).count().sort_values(['usage'], ascending=False).head(20))

    meterName = input("Enter meter name   ")

    try:
        #org_df = org_df[org_df['meterName'] == meterName]
        dfmeter= meterDict[meterName]
        print(meterName + ' dataframe')
        print(tabulate(dfmeter.head(), headers=cols, tablefmt='grid'))
    except:
        print('Hmmmm that meter does not work')

    return dfmeter, meterName

def return_measure_df(meter_df, meterName):
    measures = meter_df['measureName'].unique()
    measuresDict = {elem: pd.DataFrame() for elem in measures}
    for key in measuresDict.keys():
        measuresDict[key] = meter_df[:][meter_df.measureName == key]

    # Display Group by of meters for selected org in order to choose meter/account
    print('meter grouped by measure')
    print(meter_df.groupby(['measureName']).count().sort_values(['usage'], ascending=False).head(20))

    measureName = input("Enter measure name   ")

    try:
        dfmeasure = measuresDict[measureName]
        print(measureName + ' dataframe')
        print(tabulate(dfmeasure.head(), headers=cols, tablefmt='grid'))
    except:
        print('Hmmmm that measure does not work')

    return dfmeasure, measureName

def return_cust_df(measure_df):
    customers = measure_df['customer'].unique()
    custDict = {elem: pd.DataFrame() for elem in customers}
    for key in custDict.keys():
        custDict[key] = measure_df[:][measure_df.customer == key]

    # Display Group by of meters for selected org in order to choose meter/account
    print('usage grouped by customer')
    print(measure_df.groupby(['customer']).count().sort_values(['usage'], ascending=False).head(20))

    customer = input("Enter customer   ")

    try:
        # org_df = org_df[org_df['meterName'] == meterName]
        dfcustomer = custDict[int(customer)]
        print(customer + ' dataframe')
        print(tabulate(dfcustomer.head(), headers=cols, tablefmt='grid'))
    except:
        print('Hmmmm that customer does not work')
    return dfcustomer, customer

def prepare_df(df):
    data = df[['usage']]
    print('data size: ' + str(len(data)))
    #print(data.head())
    return data


def main():

    # plot_data()
    orgId = input("Enter Organization ID  ")
    # # TODO convert orgID to orgName
    # # orgName =
    # # print(orgName)
    org_df = return_org_df(int(orgId))
    meter_df, meterName = return_meter_df(org_df)
    measure_df, measureName = return_measure_df(meter_df, meterName)
    customer_df, customer = return_cust_df(measure_df)
    print('dataframe for org: ' + str(org_df['org'][0]) + ' meterName: ' + meterName + ' measureName: ' + measureName + 'customer: ' + customer)
    print(tabulate(customer_df.head(), headers=cols, tablefmt='grid'))
    data = prepare_df(customer_df)

    #return data
    pass


if __name__ == "__main__":
    main()
