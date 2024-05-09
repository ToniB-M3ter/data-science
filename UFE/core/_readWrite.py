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
os.environ['ORG'] = 'onfido'
os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-onfido-332767697772-us-east-1'

global DATABUCKET
DATABUCKET = os.getenv('DATABUCKET')
ORG = os.getenv('ORG')
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
            #print(data)
            with gzip.open(data, 'rb') as f_in:
                with open(key, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
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
    @staticmethod
    def write_dict_to_textfile(fileNameBase, metaDict=None):
        storedFileName = fileNameBase + '.txt'
        logger.info(storedFileName)
        output = open("/tmp/" + storedFileName, "w")
        if metaDict:
            pass
        else:
            metaDict = {'nm': 'type',
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
        for k, v in metaDict.items():
            output.writelines(f'{k},{v}\n')
        return fileNameBase


    @staticmethod
    def gz_upload(fileNameBase, filepath):
        """write to a temp file first and then load to s3"""
        storedFileName = fileNameBase + '.txt'
        savedFileName = fileNameBase + '.gz'
        with open("/tmp/{}".format(storedFileName), 'rb') as f_in:
            with gzip.open("/tmp/{}".format(savedFileName), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                # f.write(content.encode("utf-8"))

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

    def prep_forecast_for_s3(df: pd.DataFrame, df_ids, evaluation_df):  #best_forecasts, df_ids, evaluation_w_best_model
        #df.reset_index(inplace=True)
        evaluation_df.reset_index(inplace=True)

        dashboard_cols = [
            'ts_id'  # time series unique id
            , 'meter'
            , 'measure'
            , 'account_nm'
            , 'account_cd'  # account m3ter uid
            , 'tm'  # timestamp
            , 'z'  # prediction
            , 'z0'  # lower bound of 95% confidence interval
            , 'z1'  # lower bound of 95% confidence interval
            , '.model'  # model (e.g. model_)
        ]

        df = pd.merge(df, evaluation_df[['unique_id', 'best_model']], how='left', on='unique_id')
        df_ids.columns = ['account_cd', 'account_nm', 'meter', 'measure', 'unique_id']
        df = pd.merge(df, df_ids, how='left', on='unique_id')
        df['tm'] = pd.to_datetime(df['ds']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df.rename(columns={'unique_id': 'ts_id', 'best_model_y': '.model', 'best_model-lo-95':'z0','best_model_x':'z','best_model-hi-95':'z1'}, inplace=True) #TODO make generic, read column headers?
        df.sort_values(['ts_id', 'tm'], inplace=True)
        df = df[dashboard_cols]
        return df

    def write_gz_csv_to_s3(df, filepath, key): # TODO merge with logs.write_csv_log_to_S3??
        gz_buffer = BytesIO()

        with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
            df.to_csv(TextIOWrapper(gz_file, 'utf8'), index=False)

        obj = s3.Object(DATABUCKET, filepath + key)
        obj.put(Body=gz_buffer.getvalue())
        return

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
    def write_csv_log_to_S3_old(df, data_name, path):
        #print(df.head()) # <_io.TextIOWrapper name='/tmp/eval_reformat.csv' mode='r' encoding='UTF-8'>
        # set up for logging missing accounts
        OBJECT_NAME = 'logs/{}.csv'.format(data_name)
        LAMBDA_LOCAL_TMP_FILE = '/tmp/{}.csv'.format(data_name)

        with open(LAMBDA_LOCAL_TMP_FILE, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(df)
            print(writer.writerow(df))

        s3clt.upload_file(LAMBDA_LOCAL_TMP_FILE, DATABUCKET, OBJECT_NAME)

        # try with put_object
        with open(LAMBDA_LOCAL_TMP_FILE, 'r') as fd:
            print(fd)
            result = s3.put_object(
                Bucket=DATABUCKET,
                Key=OBJECT_NAME,
                Body=fd
            )

        if result['ResponseMetadata']['HTTPStatusCode'] == 200:
            response = "https://{0}.s3.us-east-1.amazonaws.com/{1}".format(DATABUCKET, OBJECT_NAME)
            logger.info(response)
        else:
            response = False

        return

    def write_csv_log_to_S3(df, data_name, path):
        OBJECT_NAME = path+'{}.csv'.format(data_name)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer)
        result = s3.Object(DATABUCKET, OBJECT_NAME).put(Body=csv_buffer.getvalue())
        if result['ResponseMetadata']['HTTPStatusCode'] == 200:
            response = "https://{0}.s3.us-east-1.amazonaws.com/{1}".format(DATABUCKET, OBJECT_NAME)
            logger.info(response)
        else:
            logger.error(result)
