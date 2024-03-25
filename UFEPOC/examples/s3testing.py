import csv
import os
import json
from pandas import util
import gzip
import shutil
import boto3
from datetime import datetime as dt
from tempfile import TemporaryFile
from io import BytesIO, StringIO, TextIOWrapper
import pandas as pd

os.environ['DATABUCKET'] = 'm3ter-usage-forecasting-poc-onfido-332767697772-us-east-1'
#'m3ter-usage-forecasting-poc-demo-332767697772-us-east-1' #'m3ter-usage-forecasting-poc-onfido-332767697772-us-east-1'
os.environ['UPLOADBUCKET'] = 'tmbbucket'
global DATABUCKET
global UPLOADBUCKET
DATABUCKET = os.getenv('DATABUCKET')
UPLOADBUCKET = os.getenv('UPLOADBUCKET')
boto3.setup_default_session(profile_name='ml-labs-prod') #ml-alpha-admin m3ter-ml-labs-prod

s3 = boto3.resource('s3')
s3client = boto3.client("s3")
bucket = s3.Bucket(UPLOADBUCKET)

def download_datapath():
    filepath = '2_tidy/1h/'  # allow selection of data
    metakey = 'usage_meta.gz' #'hier_2024_03_04_usage_meta.gz'
    key = 'usage.gz'
    return filepath, key, metakey

def upload_datapath():
    filepath = '/UFEPOC/output_files/'  # allow selection of data
    metakey = 'hier_2024_03_04_usage_meta.gz'
    key = 'engine_p.csv'
    return filepath, key, metakey

def write_csv_to_bytes(df):
    # with BytesIO() as buffer:
    #     sb = TextIOWrapper(buffer, 'utf-8', newline='')
    #     csv.writer(sb).writerows(rows)
    #     sb.flush()
    #     buffer.seek(0)

    df = pd.read_csv('/UFEPOC/data/dfUsage.csv')
    df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%d %H:%M:%S')
    df= df[0:100000]
    tmp_csv=df.to_csv()
    str_obj = StringIO(tmp_csv)
    bytes_obj = BytesIO(str_obj.encode('utf-8'))
    sb = bytes_obj
    return sb

def upload_gzipped(bucket, key, fp, compressed_fp=None, content_type='text/plain'):
    """Compress and upload the contents from fp to S3.
    If compressed_fp is None, the compression is performed in memory.
    """
    if not compressed_fp:
        compressed_fp = BytesIO()
    with gzip.GzipFile(fileobj=compressed_fp, mode='wb') as gz:
        shutil.copyfileobj(fp, gz)
    compressed_fp.seek(0)
    bucket.upload_fileobj(
        compressed_fp,
        key,
        {'ContentType': content_type, 'ContentEncoding': 'gzip'})
    return

def download_gzipped(bucket, key, fp, compressed_fp=None):
    """Download and uncompress contents from S3 to fp.
    If compressed_fp is None, the compression is performed in memory.
    """
    if not compressed_fp:
        compressed_fp = BytesIO()
    bucket.download_fileobj(key, compressed_fp)
    compressed_fp.seek(0)
    with gzip.GzipFile(fileobj=compressed_fp, mode='rb') as gz:
        shutil.copyfileobj(gz, fp) # gz is gzip file

def upload_file_in_memory(bucket, key):
    upload_file = '/UFEPOC/output_files/engine_p.csv'
    key = 'engine_p.gz'
    """In memory compression"""
    with open(upload_file, 'rb') as fp:
        upload_gzipped(bucket, key, fp)

    #with open(key, 'wb') as fp:
        #download_gzipped(bucket, key, fp)
    return

def upload_dataframe_in_memory(df, bucket, key):
    gz_buffer = BytesIO()

    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        df.to_csv(TextIOWrapper(gz_file, 'utf8'), index=False)

    s3_object = s3.Object(bucket, key)
    s3_object.put(Body=gz_buffer.getvalue())
    return

def example2(bucket, key):
    upload_file = '/UFEPOC/output_files/engine_p.csv'
    key = 'whelper_engine_p.gz'
    """Using a temporary file for compression"""
    with open(upload_file, 'rb') as fp, TemporaryFile() as helper_fp:
        upload_gzipped(bucket, key, fp, compressed_fp=helper_fp)

    with open(key, 'wb') as fp, TemporaryFile() as helper_fp:
        download_gzipped(bucket, key, fp, compressed_fp=helper_fp)

def read_from_S3(filepath, key):
    #logger = logging.getLogger('ts_engine.readWrite.read_from_s3')
    #logger.info('get file %s from %s' % (key, filepath))
    print(DATABUCKET, filepath, key)
    obj = s3.Object(DATABUCKET, filepath + key)
    with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
        data = gzipfile.read()
        data_str = data.decode()
    return data_str

def meta_str_to_dict(metadata_str):
    meta_dict={}
    dimkey_list=[]

    meta_tmp = metadata_str.split('\n')
    for i in meta_tmp:
        if len(i.split(","))==2:
            meta_dict[i.split(",")[0]]=i.split(",")[1]

    for k,v in meta_dict.items():
        if v == 'dim':
            dimkey_list.append(k)

    return dimkey_list

def get_data(filepath, metadatakey):
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
    print(metadata_str)
    return metadata_str

def write_dict_to_textfile(metaDict=None):
    storedFileNameBase = "best_usage_meta".format(dt.today().strftime("%Y_%d_%m"))
    storedFileName = "{}".format(dt.today().strftime("%Y_%d_%m"))+'_'+storedFileNameBase+'.txt'
    output = open("/tmp/" + storedFileName, "w")
    if metaDict:
        pass
    else:
        metaDict = {'nm,':'type',
                    'ts_id': 'ts_id',
                    'account_cd': 'dim',
                    'account_nm': 'dim',
                    'meter': 'dim',
                    'measure': 'dim',
                    'tm': 'time',
                    'z': 'measure',
                    'z0': 'measure',
                    'z1': 'measure',
                    '.model': 'dim',
                    '_intrvl': '1D'}
    #json.dump(testDict, open('/tmp/test.txt', 'w'))
    for k,v in metaDict.items():
        output.writelines(f'{k} {v}\n')
    return storedFileNameBase

def write_meta_tmp(storedFileNameBase):
    storedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.txt'
    savedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.gz'
    with open("/tmp/{}".format(storedFileName), 'rb') as f_in:
        with gzip.open("/tmp/{}".format(savedFileName), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            #f.write(content.encode("utf-8"))
    return

def gz_upload(storedFileNameBase, filepath):
    #fileName = 'hier_2024_03_18_usage_meta.gz'
    savedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.gz'
    with open("/tmp/{}".format(savedFileName), 'rb') as f_in:
        gzipped_content = gzip.compress(f_in.read())
        s3client.upload_fileobj(
            BytesIO(gzipped_content),
            DATABUCKET,
            filepath+savedFileName,
            ExtraArgs={"ContentType":"text/plain", "ContentEncoding":"gzip"}
        )

def read_textfile(bucket):
    original=BytesIO(b'tmbtesting some stuff here')
    original.seek(0)
    upload_gzip(bucket,'tmbmetatest.txt',original)

    with open("/UFEPOC/data/hier_2024_03_06_usage_meta.txt", "r") as fp, TemporaryFile() as helper_fp:
        #bio = BytesIO(fp.read().encode('utf-8'))
        upload_gzip(bucket,'tmbmetatest.txt',fp, compressed_fp=helper_fp)
    return

def upload_gzip(bucket, key, fp, compressed_fp=None, content_type='text/plain'):
    #if not compressed_fp:
     #       compressed_fp = BytesIO()

    with gzip.GzipFile(fileobj=compressed_fp, mode='wb') as gz:
        shutil.copyfileobj(fp, gz)
    compressed_fp.seek(0)
    print(type(compressed_fp))

    bucket.upload_fileobj(
        compressed_fp,
        bucket,
        key,
        {'ContentType': content_type, 'ContentEncoding': 'gzip'})
    return

def write_meta_to_s3(metadata_str, filepath, key):
    metadata_byte = metadata_str.encode()
    print(type(metadata_byte))
    with open('/UFEPOC/data/hier_2024_03_04_usage_meta.txt', 'rb') as f_in:
        with gzip.open('/UFEPOC/data/hier_2024_03_18_usage_meta.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    #with gzip.open('/tmp/tmpmeta.txt.gz', 'wb') as f:
     #       f.write(meta_bytes_daily)

    #with gzip.open('/tmp/tmpmeta.txt.gz', 'rb') as f:
        # obj = s3.Object(DATABUCKET, filepath + key)
        # obj.put(Body=f)

    s3client.put_object(Bucket=DATABUCKET, Body=f_out, Key=filepath + key, ContentType='text/plain', ContentEncoding='gzip')
    return

def main():
    # Get file location and name
    filepath, metadatakey, key = upload_datapath()
    #download_gzipped(bucket, filepath+key, ungzipped)
    #df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFEPOC/output_files/engine_p.csv')
    #upload_dataframe_in_memory(df, 'tmbbucket', 'csv_upload.gz')
    #example2(bucket, filepath + key)

    storedFileNameBase = write_dict_to_textfile()  # pass metaDict if created, e.g. metadata_str
    write_meta_tmp(storedFileNameBase)
    gz_upload(storedFileNameBase, '4_forecast/1D/')
    #get_data('config' + '/', 'hier_2024_03_18_usage_meta.gz')
    return

if __name__ == "__main__":
    main()