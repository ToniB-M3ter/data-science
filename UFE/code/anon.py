import pandas as pd
from faker import Faker

import readWriteS3 as rs3

def get_keys():
    metadatakey = 'usage_meta.gz'
    key = 'usage.gz'
    return key, metadatakey

def test_faker():
    fake = Faker()
    for i in range(0,10):
        print("_".join([fake.name().split(" ")[0], fake.name().split(" ")[1]])
              )

    names = [fake.unique.company() for i in range(200)]
    print(names[0:50])
    for i in range(len(names)):
         names[i] = "_".join([names[i].split(" ")[0], names[i].split(" ")[1]])
    print(names[0:50])

    assert len(set(names)) == len(names)


if __name__ == "__main__":
    freq = '1D'
    dataloadcache = pd.DataFrame()
    key, metadatakey = get_keys()
    dataloadcache, metadata_str = rs3.get_data('2_tidy/' + freq + '/', key, metadatakey)
    test_faker()