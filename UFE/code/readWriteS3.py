import gzip, csv, os
import pandas as pd
import pickle
import boto3
from io import StringIO, BytesIO, TextIOWrapper
from botocore.exceptions import ClientError
from tabulate import tabulate

os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1'
os.environ['WRITE_BUCKET'] = 'tmbbucket'
global DATABUCKET
DATABUCKET = os.getenv('DATABUCKET')
WRITE_BUCKET = os.getenv('WRITE_BUCKET')
boto3.setup_default_session(profile_name='m3ter-ml-labs-prod') #ml-alpha-admin
# Remove line and let boto3 find the role and set up default session when releasing

# Set service to s3
s3 = boto3.resource("s3")

def datapath():
    filepath = '2_tidy/1h/'  # allow selection of data
    metakey = 'usage_meta.gz'
    key = 'usage.gz'
    return filepath, metakey, key

def write_csv_to_s3(df, filepath, key):
    gz_buffer = BytesIO()

    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        df.to_csv(TextIOWrapper(gz_file, 'utf8'), index=False)

    obj = s3.Object(DATABUCKET, filepath+key)
    obj.put(Body=gz_buffer.getvalue())
    return

def write_meta_to_s3(file, filepath, key):

    gz_buffer = StringIO
    with gzip.GzipFile(filename=key+'meta.gz', compresslevel= 9, mode='w') as gz_file:
        gz_buffer

    obj = s3.Object(DATABUCKET, filepath+key)
    obj.put(Body=key+'meta.gz')
    return

def write_model_to_s3(model, filepath, key):
    with open('model.pkl', 'wb') as f:
        #pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_byte_obj = pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        obj = s3.Object(DATABUCKET, filepath+key)
        obj.put(Body=pickle_byte_obj)
    return

def read_model_from_s3():
    with open('/Users/tmb/PycharmProjects/data-science/UFE/output_files/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def read_from_S3(filepath, key):
    obj = s3.Object(DATABUCKET, filepath + key)
    with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
        data = gzipfile.read()
        data_str = data.decode()
    return data_str

def get_data(filepath, key, metadatakey):
    # get metadata
    metadata_dict = {}
    headers = []
    metadata_str = read_from_S3(filepath, metadatakey)
    lines = metadata_str.splitlines()[1:]
    for line in lines:
        key_value_pair = line.split(',')
        metadata_dict.update([key_value_pair])
        if key_value_pair[0].startswith('_'):  # non-headers begin with an underscore
            continue
        else:
            headers.append(key_value_pair[0])

    # get usage data
    usage_str = read_from_S3(filepath, key)
    df = pd.read_csv(StringIO(usage_str))
    df.columns=headers
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%dT%H:%M:%SZ')
    df=df.sort_values('tm', ascending=True)
    df.dropna(subset=['y'], inplace = True) # need to generisise this
    return df

def get_data_local():
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/dfUsage.csv')
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%d %H:%M:%S')
    return df

def analyse_data(df):

    #df.astype(bool).sum(axis=0)  # count non-Nan's

    counts = df.notnull().groupby(df['account']).count()
    print('Account Event Counts')
    print(tabulate(counts, headers="keys", tablefmt="psql"))

    return

def select_ts(df):
    analyse_data(df)

    # Select Account(s)
    print(str(df['account'].nunique()) + ' Unique accounts')
    all = ['all','All','ALL']
    account = input('Enter an account, all or count: ' )
    if account in all:
        #accounts = df['account'].unique()[0:30]
        accounts = ['AssembledHQ Prod',
                    'BurstSMS - Production',
                    'Burst SMS - Local Test',
                    'Sift Forecasting',
                    'Onfido Dev',
                    'Onfido Prod',
                    'Patagona - Sandbox',
                    'Patagona - Production',
                    'Regal.io Prod',
                    'm3terBilllingOrg Production',
                    'Tricentis Prod'] # subset of accounts that are known to work TODO add criteria to select ts which will work
        df = df.loc[(df['account'].isin(accounts))]
    elif account == 'count':
        print("Sample count of measurements by time-step")
        df = df.groupby(['tm']).agg({'n_loads': 'count', 'n_events': 'count'})
        # df = df.groupby(['tm', 'meter']).agg({'n_loads': 'count', 'n_events': 'count'}) # choose between loads and events
        print(tabulate(df.head(50), headers="keys", tablefmt="psql"))
        df.reset_index(names='tm', inplace=True)
    else:
        try:
            df = df.loc[(df['account'] == account)]
        except:
            print("That account doesn't exist")

    # Select meter
    print(df['meter'].unique())
    meter = input('Enter a meter? ')
    if meter in all:
        pass
    else:
        try:
            df = df.loc[df['meter'] == meter]
        except:
            print("That meter doesn't exist")

    print(str(len(df)) + ' records from ' + str(df['tm'].min()) + ' to ' + str(df['tm'].max()) )

    print(tabulate(df.head(10), headers="keys", tablefmt="psql"))
    return df

def main():
    filepath, metadatakey, key = datapath() # If calling directly get data location
    data = get_data(filepath, key, metadatakey)
    data = select_ts(data)
    return data

if __name__ == "__main__":
        main()

