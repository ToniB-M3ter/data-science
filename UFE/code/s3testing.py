import csv
import os
from pandas import util
import gzip
import shutil
import boto3
from tempfile import TemporaryFile
from io import BytesIO, StringIO, TextIOWrapper
import pandas as pd

os.environ['DATABUCKET'] = 'm3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1'
os.environ['UPLOADBUCKET'] = 'tmbbucket'
global DATABUCKET
global UPLOADBUCKET
DATABUCKET = os.getenv('DATABUCKET')
UPLOADBUCKET = os.getenv('UPLOADBUCKET')
boto3.setup_default_session(profile_name='customers-admin') #ml-alpha-admin m3ter-ml-labs-prod

s3 = boto3.resource('s3')
bucket = s3.Bucket(UPLOADBUCKET)

def download_datapath():

    filepath = '2_tidy/1h/'  # allow selection of data
    metakey = 'usage_meta.gz'
    key = 'usage.gz'
    return filepath, metakey, key

def upload_datapath():
    filepath = '/Users/tmb/PycharmProjects/data-science/UFE/output_files/'  # allow selection of data
    metakey = 'usage_meta.gz'
    key = 'engine_p.csv'
    return filepath, metakey, key

def write_csv_to_bytes(df):
    # with BytesIO() as buffer:
    #     sb = TextIOWrapper(buffer, 'utf-8', newline='')
    #     csv.writer(sb).writerows(rows)
    #     sb.flush()
    #     buffer.seek(0)

    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/dfUsage.csv')
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
    upload_file = '/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv'
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
    upload_file = '/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv'
    key = 'whelper_engine_p.gz'
    """Using a temporary file for compression"""
    with open(upload_file, 'rb') as fp, TemporaryFile() as helper_fp:
        upload_gzipped(bucket, key, fp, compressed_fp=helper_fp)

    with open(key, 'wb') as fp, TemporaryFile() as helper_fp:
        download_gzipped(bucket, key, fp, compressed_fp=helper_fp)


def main():
    # Get file location and name
    filepath, metadatakey, key = upload_datapath()
    ungzipped = BytesIO()

    #download_gzipped(bucket, filepath+key, ungzipped)

    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')
    upload_dataframe_in_memory(df, 'tmbbucket', 'csv_upload.gz')

    #example2(bucket, filepath + key)


    return

if __name__ == "__main__":
    main()