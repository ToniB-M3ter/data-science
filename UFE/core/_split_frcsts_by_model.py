from datetime import datetime as dt
import pandas as pd
import _readWrite as rw

freq = '1D'
# folders
tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '4_forecast/'
combined_folder = '5_combine/'
logs_folder = 'logs/'

def get_forecasts():
    df = pd.read_csv('/Users/tmb/PycharmProjects/data-science/UFE/output_files/onfido/all_forecasts-dfaa0447-92fd-4d3c-8c72-f816ad4d0f7f.csv',
                    # dtype={"tm": datatime, "username": "string"},
                    parse_dates=['ds'], index_col=0)
    return df

def split(df: pd.DataFrame) ->dict:
    other_cols = ['ds']
    lo_cols = [x for x in df.columns if 'lo' in x]
    hi_cols = [x for x in df.columns if 'hi' in x]
    base_cols = [y for y in df if y not in lo_cols and y not in hi_cols and y not in other_cols]

    modelsdict = {}
    for i in base_cols:
        # Need special case for Naive/SeasonalNaive
        if i =='Naive':
            dfcols = [x for x in df.columns if i in x and 'SeasonalNaive' not in x]
        else:
            dfcols = [x for x in df.columns if i in x]

        dfcols.extend(other_cols)
        modelsdict[i] = df[dfcols]
        modelsdict[i].columns = ['tm', 'z', 'z0','z1']
        modelsdict[i] = modelsdict[i].reset_index().rename(columns={'unique_id':'ts_id'})
        #print(modelsdict[i].head())
    return modelsdict

def prep_fcst_by_model_for_save(modelsdict: dict):
    metaDict = {'nm': 'type',
                'unique_id': 'ts_id',
                'tm': 'time',
                'z': 'measure',
                'z0': 'measure',
                'z1': 'measure',
                '_intrvl': '1D'}

    for k,v in modelsdict.items():
        rw.tsdata.write_gz_csv_to_s3(v, forecast_folder + '1D' + '/', k + '_'+ dt.today().strftime("%Y_%m_%d") + '_usage.gz')
        fileNameBase = "{}_{}_usage_meta".format(k, dt.today().strftime("%Y_%m_%d"))
        fileNameBase = rw.metadata.write_dict_to_textfile(fileNameBase)  # pass metaDict if created, e.g. metadata_str
        rw.metadata.gz_upload(fileNameBase, forecast_folder + freq + '/')

def main(): # can be run separately picking up local file
    allforecasts = get_forecasts()
    modelsdict = split(allforecasts)
    prep_fcst_by_model_for_save(modelsdict)

if __name__ == "__main__":
    main()