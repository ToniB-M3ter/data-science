import gzip, csv, os
import pandas as pd
import pickle
import shutil
import boto3
from io import StringIO, BytesIO, TextIOWrapper
from botocore.exceptions import ClientError

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 25)

# Remove line and let boto3 find the role and set up default session when releasing
boto3.setup_default_session(profile_name='ml-labs-prod') #ml-alpha-admin ml-labs-prod    ml-alpha-admin
os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-onfido-332767697772-us-east-1' #'m3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1'

global DATABUCKET
DATABUCKET = os.getenv('DATABUCKET')
USER = os.getenv('USER')

import logging
module_logger = logging.getLogger('ts_engine.readWrite')
logger = logging.getLogger('ts_engine.readWrite')

# Set service to s3
s3clt = boto3.client('s3')
s3 = boto3.resource("s3")
s3client = boto3.client("s3")

def datapath(freq):
    filepath = '2_tidy/' + freq  # allow selection of data
    metakey = 'hier_2024_03_04_usage_meta.gz'
    key = 'usage.gz'
    return filepath, metakey, key

def read_from_S3(filepath, key):
    #logger = logging.getLogger('ts_engine.readWrite.read_from_s3')
    #logger.info('get file %s from %s' % (key, filepath))
    print(DATABUCKET, filepath, key)
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
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%dT%H:%M:%SZ' )
    df=df.sort_values('tm', ascending=True)
    #df.dropna(subset=['y'], inplace = True) # need to generalise this
    return df, metadata_str

def get_data_local():
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/dfUsage.csv')
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%d %H:%M:%SZ')
    return df

def read_model_from_s3(filepath, key):  #(fit_folder+freq, model_aliases[0]+'.pkl')
    model = pickle.loads(s3.Bucket(DATABUCKET).Object(filepath + key).get()['Body'].read())
    return model

def read_model_from_local():
    with open('/Users/tmb/PycharmProjects/data-science/UFE/output_files/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def write_model_to_s3(model, filepath, key):
    serialised_model = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    response = boto3.client('s3').put_object(
        Body=serialised_model,
        Bucket=DATABUCKET,
        Key=filepath+key
    )
    return

def write_csv_log_to_S3(df, data_name):
    # set up for logging missing accounts
    OBJECT_NAME ='logs/{}.csv'.format(data_name)
    LAMBDA_LOCAL_TMP_FILE = '/tmp/{}.csv'.format(data_name)

    with open(LAMBDA_LOCAL_TMP_FILE, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(df)

    s3clt.upload_file(LAMBDA_LOCAL_TMP_FILE, DATABUCKET, OBJECT_NAME)
    return

def write_meta_tmp(fileName):
    """write meta data to file to be uploaded to s3"""
    with open('/tmp/{}.txt'.format(fileName), 'rb') as f_in:
        with gzip.open('/Users/tmb/PycharmProjects/data-science/UFE/data/hier_2024_03_06_usage_meta.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return

def write_meta_to_s3(metadata_str, freq, filepath, key):
    metadata_byte = metadata_str.encode()

    meta_bytes_daily = \
    b"""nm, typ
    ts_id,  ts_id
    account_cd, dim
    account_nm, dim
    meter, dim
    measure, dim
    tm, time 
    z, measure 
    z0, measure
    z1, measure
    .model, dim
    _intrvl, 1D"""

    meta_bytes_hourly = \
        b"""nm, typ
        ts_id,  ts_id
        account_cd, dim
        account_nm, dim
        meter, dim
        measure, dim
        tm, time 
        z, measure 
        z0, measure
        z1, measure
        .model, dim
        _intrvl, 1h"""

    if freq == '1h':
        with gzip.open('/tmp/tmpmeta.txt.gz', 'wb') as f:
            f.write(meta_bytes_hourly)
    elif freq == '1D':
        with gzip.open('/tmp/tmpmeta.txt.gz', 'wb') as f:
            f.write(meta_bytes_daily)
    else:
        print('No associated freq for metadata file')

    with gzip.open('/tmp/tmpmeta.txt.gz', 'rb') as f:
        # obj = s3.Object(DATABUCKET, filepath + key)
        # obj.put(Body=f)
        s3client.put_object(Bucket=DATABUCKET, Body=f, Key=filepath + key, ContentType='text/plain', ContentEncoding='gzip')
    return

def write_gz_csv_to_s3(df, filepath, key):
    gz_buffer = BytesIO()

    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        df.to_csv(TextIOWrapper(gz_file, 'utf8'), index=False)

    obj = s3.Object(DATABUCKET, filepath+key)
    obj.put(Body=gz_buffer.getvalue())
    return

def main():
    freq_input = input("Hourly (1h) or Daily (1D) frequency: ")
    freq = freq_input + '/'
    filepath, metadatakey, key = datapath(freq) # If calling directly get data location
    data = get_data(filepath, key, metadatakey)
    return data

if __name__ == "__main__":
        main()

