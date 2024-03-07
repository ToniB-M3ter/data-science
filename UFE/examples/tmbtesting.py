# running code from CL
# poetry run python UFE/code/tmbtesting.py

import sys
sys.path.insert(1, '/Users/tmb/PycharmProjects/data-science/UFE/code')
# plotting
import matplotlib.pyplot as plt
import boto3
from faker import Faker
s3 = boto3.resource("s3")
s3client = boto3.client("s3")
import sys, os
import pickle
import gzip
from io import StringIO, BytesIO, TextIOWrapper
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

import readWriteS3 as rs3

# predicting
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

os.environ['DATABUCKET']  = 'm3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1'
DATABUCKET = os.getenv('DATABUCKET')
boto3.setup_default_session(profile_name='ml-labs-prod')

df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/df.csv', index_col=0)

# Set service to s3
s3 = boto3.resource("s3")

def plot_raw(data: pd.DataFrame, title: str):
    fig, ax = plt.subplots()
    ax = data.plot(ax=ax).set(title=title)
    plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output/{}.jpg'.format(title[0:7]))
    plt.show()
    #fig = px.line(data, x=data.index, y=data['Volume'], title='Yahoo Stock')
    #fig.show()
    return

def plot_forecast(res, title):
    fig, ax = plt.subplots()
    fig = res.plot_predict(720,840)
    fig.suptitle(title)
    plt.savefig('/Users/tmb/PycharmProjects/data-science/UFE/output/{}.png'.format(title[0:7]))
    #res.predict().plot(ax=ax)
    plt.show()
    return

def get_housing_data():
    data = pdr.get_data_fred("HOUSTNSA", "1959-01-01", "2019-06-01")
    title= 'Percent Change in Housing Prices\nFrom 1959-01-01 to 2019-06-01'
    housing = data.HOUSTNSA.pct_change().dropna()
    # Scale by 100 to get percentages
    housing = 100 * housing.asfreq("MS")
    plot_raw(housing, title)
    return data, title

def get_nixtla_data():
    Y_df = pd.read_parquet('https://datasets-nixtla.s3.amazonaws.com/m4-hourly.parquet')
    print(Y_df.info())
    Y_df.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/data/nixtla_sample_data.csv')
    return

def get_engine_data():
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/engine_p.csv')
    df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d %H:%M:%S')
    return df

def convert_to_dashboard_format(df: pd.DataFrame, model_aliases):
    dashboard_cols = [
        'tm'  # timestamp
        #'meter'
        #'measurement'
        #'account'
        #'account_id'  # account m3ter uid
        ,'ts_id'  # ts unique id
        ,'z'  # prediction
        ,'z0'  # lower bound of 95% confidence interval
        ,'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df'+alias, alias, alias+'-lo-95', alias+'-hi-95']
        iterator_list[0] = df[['ds','unique_id', iterator_list[1], iterator_list[2], iterator_list[3] ]]
        iterator_list[0]['.model'] = alias
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)

    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/forecasts.csv')
    return dfAll

def smoothing(data):
    fit1 = SimpleExpSmoothing(data[0:200]).fit(smoothing_level=0.2, optimized=False)
    fit2 = SimpleExpSmoothing(data[0:200]).fit(smoothing_level=0.8, optimized=False)
    plt.figure(figsize=(18, 8))
    plt.title('Smoothing Levels 0.2 vs 0.8')
    plt.plot(data[0:200], marker='o', color="black", label="original")
    plt.plot(fit1.fittedvalues, marker="o", color="b", label="0.2")
    plt.plot(fit2.fittedvalues, marker="o", color="r", label="0.8")
    plt.legend()
    plt.xticks(rotation="vertical")
    plt.show()
    return

def decompose(data):
    print(data.describe())
    print(data.head())
    stl = STL(data, period='M', seasonal=7)
    res = stl.fit()
    fig = res.plot()
    return

def fit_data(data):
    # select model
    mod = AutoReg(data, 3, old_names=False)
    # fit model
    res = mod.fit()
    print(res.summary())


    # with covariance estimators as OLS
    #res2 = mod.fit(cov_type='HC0')
    #print(res2.summary())
    # select model order
    sel = ar_select_order(data, 13, old_names=False)
    sel.ar_lags
    res = sel.model.fit()
    print(res.summary())
    return res

def write_model_to_s3(model, filepath, key):
    with open('model.pkl', 'wb') as f:
        #pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        bytes_output = BytesIO()
        pickle_byte_obj = pickle.dump(model, bytes_output, protocol=HIGHEST_PROTOCOL)
        #pickle_byte_obj = pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        obj = s3.Object(DATABUCKET, filepath+key)
        obj.put(Body=pickle_byte_obj)
    return

def write_model_to_s3_2(model):
    filepath = 'UFE/models/'
    key = 'tmbtest_model.pkl'

    # serialise and write to temp file
    with open('/UFE/output_files/model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # open temp file
    with open('/UFE/output_files/model.pkl', 'rb') as f:
        # write
        response = boto3.client('s3').put_object(
            Body=f,
            Bucket='m3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1',
            Key='4_fit/1h/tmbtest_model.pkl'
        )
    return

def read_model_from_s3(filepath, key):
    s3 = boto3.resource('s3')
    model = pickle.loads(s3.Bucket(DATABUCKET).Object(filepath + key).get()['Body'].read())
    print(type(model))
    return model

def prep_data_for_s3():
    model_aliases = ['AE']
    dashboard_cols = [
        'tm'  # timestamp
        , 'meter'
        , 'measurement'
        , 'account_id'  # account m3ter uid
        , 'account'
        # , 'ts_id'  # ts unique id
        , 'z'  # prediction
        , 'z0'  # lower bound of 95% confidence interval
        , 'z1'  # lower bound of 95% confidence interval
        , '.model'  # model (e.g. model_)
    ]

    df_ids = pd.read_csv('/UFE/output_files/df_ids.csv')
    df = pd.read_csv('/UFE/output_files/only_forecst.csv')

    pat = "|".join(df_ids.account)
    df.insert(0, 'account', df['unique_id'].str.extract("(" + pat + ')', expand=False))
    df = df.merge(df_ids[['meter','measurement', 'account_id', 'account']], on='account')
    df = df[['ds','meter','measurement', 'account_id', 'account', 'AE', 'AE-lo-95', 'AE-hi-95']]

    dfs = []

    for alias in model_aliases:
        iterator_list = ['df' + alias, alias, alias + '-lo-95', alias + '-hi-95']
        iterator_list[0] = df[['ds', 'meter', 'measurement', 'account_id', 'account', iterator_list[1], iterator_list[2], iterator_list[3]]]
        iterator_list[0]['.model'] = alias
        iterator_list[0].columns = dashboard_cols
        dfs.append(iterator_list[0])

    dfAll = pd.concat(dfs, ignore_index=True)

    dfAll.to_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/all_forecasts_test.csv')

def prep_meta_data_for_s3_new():
    BUCKET = 'm3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1',
    gzbuffer = StringIO()

    # Set service to s3
    s3 = boto3.resource("s3")

    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())

    writer = gzip.GzipFile(None, 'w', 6, gzbuffer)
    for line in meta_list:
        writer.write(','.join(map(str, line))+'\n')
    writer.close()
    #file = gzip.open('/Users/tmb/PycharmProjects/data-science/UFE/output_files/tmbmeta.gz', 'wt')
    #for line in meta_list:
    #    file.write(','.join(map(str, line))+'\n')
    #file.close()
    # response = boto3.client('s3').put_object(
    #     Body=gzbuffer.getvalue(),
    #     Bucket='m3ter-usage-forecasting-poc-m3ter-332767697772-us-east-1',
    #     Key='4_fit/1h/tmbmeta.gz'
    # )
    return

def prep_meta_data_for_s3():
    # TODO change meta as dictionary passed parameter to function
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())
    with open ('/UFE/output_files/tmbmeta.gz', 'w') as file: # TODO change to local temp folder
        for i in meta_list:
            file.write(','.join(map(str, i))+'\n')  # file type => _io.TextIOWrapper
    return file

def write_meta_to_s3(file, filepath, key):
    meta = {'nm': 'typ', 'meter': 'dim', 'measurement': 'dim', 'account': 'dim', 'account_id': 'dim', '.model': 'dim', 'z': 'measure', 'tm': 'time', '_intrvl': '1h', 'z0': 'measure', 'z1': 'measure'}
    meta_list = list(meta.items())

    gz_buffer = BytesIO
    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        for line in meta_list:
            gz_file.write((','.join(map(str, line)) + '\n').encode())
            # bytes() is an easy way to convert StringIO()'s .getvalue() string to bytes! Then you can gzip it.
    print(type(gz_file))
    obj = s3.Object(DATABUCKET, filepath+key)
    obj.put(Body=gz_buffer.get_value())
    return

def only_bytes(filepath, key):
    s3client = boto3.client("s3")
    meta_bytes = b"""nm, typ
    meter, dim
    measurement, dim
    account, dim
    account_id, dim
    model, dim
    z, measure 
    tm, time 
    _intrvl, 1h
    z0, measure
    z1, measure"""

    with gzip.open('/UFE/output_files/file.txt.gz', 'wb') as f:
        f.write(meta_bytes)

    with gzip.open('/UFE/output_files/file.txt.gz', 'rb') as f:
        #obj = s3.Object(DATABUCKET, filepath + key)
        #obj.put(Body=f)
        s3client.put_object(Bucket=DATABUCKET, Body=f, Key=filepath+key)
    return

def bytes_with_buffer(filepath, key):
    meta_bytes = b"""nm, typ
    meter, dim
    measurement, dim
    account, dim
    account_id, dim
    model, dim
    z, measure 
    tm, time 
    _intrvl, 1h
    z0, measure
    z1, measure"""

    buff = BytesIO()
    with gzip.GzipFile(fileobj=buff, mode='wb') as g:
       g.write(buff.getvalue())
       buff.seek(0)
    obj = s3.Object(DATABUCKET, filepath + key)
    obj.put(Body=buff.getvalue())

    s3client.upload_fileobj(Bucket=DATABUCKET, Fileobj=buff, Key=filepath+key)

def only_strings(filepath, key):
    meta_bytes=b"""nm, typ
    meter, dim
    measurement, dim
    account, dim
    account_id, dim
    model, dim
    z, measure 
    tm, time 
    _intrvl, 1h
    z0, measure
    z1, measure"""

    obj = s3.Object(DATABUCKET, filepath + key)
    obj.put(Body=meta_bytes)

def write_csv_to_s3(df, filepath, key):
    gz_buffer = BytesIO()

    with gzip.GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
        df.to_csv(TextIOWrapper(gz_file, 'utf8'), index=False)

    obj = s3.Object(DATABUCKET, filepath+key)
    obj.put(Body=gz_buffer.getvalue())
    return

def meta_str_to_dict(metadata_str):
    meta_dict={}
    dimkey_list=[]
    freq = '1D'
    dataloadcache= pd.DataFrame()
    metadatakey = 'hier_2024_03_04_usage_meta.gz'
    key = 'usage.gz'
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    print(metadata_str.split('\n'))
    test = metadata_str.split('\n')
    for i in test:
        if len(i.split(","))==2:
            meta_dict[i.split(",")[0]]=i.split(",")[1]


    for k,v in meta_dict.items():
        if v == 'dim':
            dimkey_list.append(k)

    print(dimkey_list)
    return dimkey_list, meta_dict

def main():
    #get_nixtla_data()
    #df = get_engine_data()
    #model_aliases=['AE', 'SN']
    #convert_to_dashboard_format(df, model_aliases)
    #data, title = get_housing_data()
    #heatmap()
    #smoothing(data)
    #decompose(data)
    #res = fit_data(data)
    #plot_forecast(res, title)

    ################################################
    # compressing, uncompressing, reading, writing to s3
    #read_model_from_s3('4_fit/1h/', 'Prompt_model.pkl')

    meta_str_to_dict()
    #test_faker()


def test_faker():
    fake = Faker()
    for i in range(0,10):
        print("_".join([fake.name().split(" ")[0], fake.name().split(" ")[1]])
              )

    #prep_meta_data_for_s3_new()
    #metadatafile = prep_meta_data_for_s3()
    #write_meta_to_s3(metadatafile, '4_fit/1h/', 'tmbmeta.gz')
    #only_strings('4_fit/1h/', 'tmbmeta.txt')
    #only_bytes('4_fit/1h/', 'tmbmeta.gz')

    return


if __name__ == "__main__":
    main()


######################################################################################################################
# Deprecated functions
#
######################################################################################################################

# def make_forecasts(df: DataFrame, horizon, season, window):
#     print('horizon ' + str(horizon) + '  season ' + str(season) + ' window ' + str(window))
#     # prep data / choose train
#     train = df.iloc[:horizon]
#     valid = df.iloc[horizon + 1:]
#     h = round((valid['ds'].nunique()) * 0.5)  # prediction 50% of the valid set's time interval into the future
#
#     # plot the training data
#     plot_HW_train(train, df['unique_id'].iloc[0], 'train')
#
#     # Set up models to fit: e.g. additive (HW_A) & multiplicative (HW_M)
#     model_aliases = [
#         'SN',
#         'N',
#         # 'HA',
#         'AA',
#         'AE',
#         'HW_M',
#         'HW_A'
#     ]
#
#     ts_models = [
#         SeasonalNaive(season_length=season, alias='SN'),
#         Naive(alias='N'),
#         # HistoricAverage(alias='HA'),
#         AutoARIMA(season_length=season, alias='AA'),
#         AutoETS(model=['Z', 'Z', 'Z'], season_length=season, alias='AE'),
#         HoltWinters(season_length=season, error_type='M', alias='HW_M'),
#         HoltWinters(season_length=season, error_type='A', alias='HW_A')
#     ]
#
#     model = StatsForecast(models=ts_models,
#                           freq='H',
#                           n_jobs=-1)
#
#     # fit model and record time
#     init = time()
#     model.fit(train)
#     end = time()
#     print(f'Forecast Minutes: {(end - init) / 60}')
#
#     # predict future h periods, with 95% confidence level
#     p = model.predict(h=h, level=[95])
#     p = p.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')
#
#     # plot predictions
#     plot_HW_forecast_vs_actuals(p, model_aliases)
#     return



# def decompose(dfUsage: DataFrame)-> Series:
#     result = seasonal_decompose(dfUsage['y'],model='additive')
#     result.plot()
#     plt.show()
#     return
