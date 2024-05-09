from datetime import datetime as dt
import _readWrite as rw
import pandas as pd

freq = '1D'
# folders
tidy_folder = '2_tidy/'
fit_folder = '4_fit/'
forecast_folder = '4_forecast/'
logs_folder = 'logs/'

def save(f):
    rw.tsdata.write_gz_csv_to_s3(f, forecast_folder + '1D' + '/',
                             'revenue_' + dt.today().strftime("%Y_%m_%d") + '_usage.gz')
    fileNameBase = rw.metadata.write_dict_to_textfile()  # pass metaDict if created, e.g. metadata_str
    savedFileName = rw.metadata.write_meta_tmp(fileNameBase)
    rw.metadata.gz_upload(savedFileName, forecast_folder + freq + '/')

if __name__ == "__main__":
    f = pd.read_csv('/Users/tmb/PycharmProjects/UFE-to-Rev/output_files/allFRevenue.csv',
                    #dtype={"tm": datatime, "username": "string"},
                    parse_dates=['tm'], index_col = 0)
    save(f)