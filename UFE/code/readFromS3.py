import gzip, csv, os
import pandas as pd
import boto3
from io import StringIO
from botocore.exceptions import ClientError
from tabulate import tabulate

os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-demo-501098594448-eu-west-2'
os.environ['WRITE_BUCKET'] = 'tmbbucket'
global DATABUCKET
DATABUCKET = os.getenv('DATABUCKET')
WRITE_BUCKET = os.getenv('WRITE_BUCKET')
boto3.setup_default_session(profile_name='ml-alpha-admin') # Remove line and let boto3 find the role and set up default session when releasing

# Set service to s3
s3 = boto3.resource("s3")

def datapath():
    filepath = '2_tidy/1h/'  # allow selection of data
    metakey = 'usage_meta.gz'
    key = 'usage.gz'
    return filepath, metakey, key

def write_to_S3(data, data_name):
    # set up for logging missing accounts
    OBJECT_NAME ='logs/{}.txt'.format(data_name)
    LAMBDA_LOCAL_TMP_FILE = '/tmp/{}.txt'.format(data_name)
    with open(LAMBDA_LOCAL_TMP_FILE, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    s3.upload_file(LAMBDA_LOCAL_TMP_FILE, WRITE_BUCKET, OBJECT_NAME)
    return

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
    #print(tabulate(df.head(10), headers="keys" , tablefmt="psql")) # optional print of df
    return df

def select_ts(df):
    print(df.astype(bool).sum(axis=0)) # count non-Nan's

    print('Unique accounts' + str(df['account'].unique()))
    all = ['all','All','ALL']
    account = input('Enter an account or enter All: ' )
    if account in all:
        # find time series for all accounts
        meter = input('Enter a meter or enter All: ')
        if meter in all:
            # time series everything tha makes sense
            pass
        pass
    else:
        try:
            df = df.loc[(df['account'] == account)]
        except:
            print("That account doesn't exist")
        print("Account's Unique meters")
        print(df['meter'].unique())
        meter = input('Enter a meter or enter All? ')
        if meter in all:
            pass
        else:
            try:
                df = df.loc[(df['account']==account) & (df['meter']==meter)]
            except:
                print("That meter doesn't exist")

    print(str(len(df)) + ' records from ' + str(df['tm'].min()) + ' to ' + str(df['tm'].max()) )

    print(tabulate(df.head(10), headers="keys", tablefmt="psql"))
    return df

def main():
    filepath, metadatakey, key = datapath()
    data = get_data(filepath, key, metadatakey)
    data = select_ts(data)
    return data

if __name__ == "__main__":
        main()

