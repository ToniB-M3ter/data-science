import gzip, csv, os, sys, io
import pickle
import shutil
import boto3
import logging
import pandas as pd
from datetime import datetime as dt
from io import StringIO, BytesIO, TextIOWrapper

module_logger = logging.getLogger('UFE.readWrite')
logger = logging.getLogger('URE.readWrite')

############################################# Set parameters #############################################
# Remove line and let boto3 find the role and set up default session when releasing
boto3.setup_default_session(profile_name='ml-labs-prod')
# TODO: Parameters to be set with config files
os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-onfido-332767697772-us-east-1'

global DATABUCKET
DATABUCKET = os.getenv('DATABUCKET')
USER = os.getenv('USER')
#################################################################################################################

# Set service to s3
s3clt = boto3.client('s3')
s3 = boto3.resource("s3")
s3client = boto3.client("s3")

class common_actions():
    def read_from_S3(filepath, key):
        # logger = logging.getLogger('ts_engine.readWrite.read_from_s3')
        # logger.info('get file %s from %s' % (key, filepath))
        print('read_from S3')
        print(DATABUCKET, filepath, key)
        obj = s3.Object(DATABUCKET, filepath + key)
        try:
            with gzip.GzipFile(fileobj=obj.get()["Body"]) as gzipfile:
                data = gzipfile.read()
                data_str = data.decode()
        except:
            # TODO fix get metadata routine
            body = obj.get()["Body"]
            file_like_obj = io.BytesIO(body.read())
            data = gzip.GzipFile(fileobj=io.BytesIO(s3.Object(DATABUCKET, filepath + key).get()['Body'].read()),
                                 mode='rb').read()
            print('data')
            print(data)
            with gzip.open(data, 'rb') as f_in:
                with open(key, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(key)
        return data_str

class metadata():
    # reading from s3 and preparing for processing
    def get_metadata(filepath: str, metadatakey: str)-> dict:
        print('get_METAdata')
        print(DATABUCKET, filepath, metadatakey)
        # get metadata
        metadata_dict = {}
        headers = []
        metadata_str = common_actions.read_from_S3(filepath, metadatakey)
        lines = metadata_str.splitlines()[1:]
        for line in lines:
            key_value_pair = line.split(',')
            metadata_dict.update([key_value_pair])
            if key_value_pair[0].startswith('_'):  # non-headers begin with an underscore
                continue
            else:
                headers.append(key_value_pair[0])
        return metadata_str, headers

    def meta_str_to_dict(metadata_str): # TODO merge with get_metadata
        meta_dict={}
        meta_tmp = metadata_str.split('\n')
        for i in meta_tmp:
            if len(i.split(","))==2:
                meta_dict[i.split(",")[0]]=i.split(",")[1]
        return meta_dict

    def meta_to_dim_list(meta_dict):
        dimkey_list = []
        for k,v in meta_dict.items():
            if v == 'dim':
                dimkey_list.append(k)
        return dimkey_list

    # saving for later use
    def write_dict_to_textfile(metaDict=None):
        storedFileNameBase = "best_usage_meta".format(dt.today().strftime("%Y_%d_%m"))
        storedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.txt'
        output = open("/tmp/" + storedFileName, "w")
        if metaDict:
            pass
        else:
            metaDict = {'nm,': 'type',
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
        # json.dump(testDict, open('/tmp/test.txt', 'w'))
        for k, v in metaDict.items():
            output.writelines(f'{k} {v}\n')
        return storedFileNameBase

    def write_meta_tmp(storedFileNameBase):
        storedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.txt'
        savedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.gz'
        with open("/tmp/{}".format(storedFileName), 'rb') as f_in:
            with gzip.open("/tmp/{}".format(savedFileName), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                # f.write(content.encode("utf-8"))
        return

    def gz_upload(storedFileNameBase, filepath):
        # fileName = 'hier_2024_03_18_usage_meta.gz'
        savedFileName = "{}".format(dt.today().strftime("%Y_%d_%m")) + '_' + storedFileNameBase + '.gz'
        with open("/tmp/{}".format(savedFileName), 'rb') as f_in:
            gzipped_content = gzip.compress(f_in.read())
            s3client.upload_fileobj(
                BytesIO(gzipped_content),
                DATABUCKET,
                filepath + savedFileName,
                ExtraArgs={"ContentType": "text/plain", "ContentEncoding": "gzip"}
            )

    def convert_text_to_zip(self):
        with open("/tmp/{}".format('meta_txt_file.txt'), 'rb') as f_in: # probably can stay hardcoded as its a temp file
            with gzip.open("/tmp/{}".format('meta_txt_file.gz'), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                #f.write(content.encode("utf-8"))

    def meta_dict_to_tmp_txt_file(self, meta_dict):
        if 'z' not in meta_dict.keys(): # if the forecasts keys are not in dictionary add them
            meta_dict['z']= 'measure'
            meta_dict['z0'] = 'measure'
            meta_dict['z1'] = 'measure'

        f_meta = open('tmp/meta_txt_file.txt', "w")
        f_meta.write("\n")
        for k in meta_dict.keys():
            f_meta.write("{}, {}\n".format(k, meta_dict[k]))
        f_meta.close()

        self.convert_text_to_zip() # convert tmp file to .gz for loading to s3

class tsdata():
    def get_data(filepath, key, cols):
        # get usage data
        usage_str = common_actions.read_from_S3(filepath, key)
        df = pd.read_csv(StringIO(usage_str))
        df.columns = cols
        df['tm'] = pd.to_datetime(df['tm'], format='%Y-%m-%dT%H:%M:%SZ')
        df = df.sort_values('tm', ascending=True)
        # df.dropna(subset=['y'], inplace = True)
        return df

class model():
    def read_model_from_s3(filepath, key):  # (fit_folder+freq, model_aliases[0]+'.pkl')
        model = pickle.loads(s3.Bucket(DATABUCKET).Object(filepath + key).get()['Body'].read())
        return model

    def write_model_to_s3(model, filepath, key):
        serialised_model = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        response = boto3.client('s3').put_object(
            Body=serialised_model,
            Bucket=DATABUCKET,
            Key=filepath + key
        )
        return

class logs():
    def write_csv_log_to_S3(df, data_name):  #TODO add S3 path
        # set up for logging missing accounts
        OBJECT_NAME = 'logs/{}.csv'.format(data_name)
        LAMBDA_LOCAL_TMP_FILE = '/tmp/{}.csv'.format(data_name)

        with open(LAMBDA_LOCAL_TMP_FILE, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(df)

        s3clt.upload_file(LAMBDA_LOCAL_TMP_FILE, DATABUCKET, OBJECT_NAME)
        return