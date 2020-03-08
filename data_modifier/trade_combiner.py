import os
from pprint import pprint

import numpy as np
import pandas as pd

DATA_PATH = '/home/sankalp/tkg/data/trade-data'
SAVE_PATH = '/home/sankalp/tkg/data/trade-data-cleaned'
files = os.listdir(DATA_PATH)
files.sort()


def get_country_from_filename(country):
    temp = 'Trade_Map_-_List_of_partners_markets_for_a_product_commercialized_by_'
    country = country[len(temp):].split('.')[0].split('(')[0].strip()

    if country[-1] == '_':
        country = country[:-1]

    return country


countries = list(set([get_country_from_filename(a) for a in files]))
countries.sort()
print(countries)

data = {}

print()
for file in files:
    print(file)
    country = get_country_from_filename(file)
    if country not in data:
        data[country] = None

    file_path = '{}/{}'.format(DATA_PATH, file)
    df = pd.read_csv(file_path, sep='\t', index_col=0, header=0)

    df.drop(df.columns[[-1, -2, -3]], axis=1, inplace=True)
    if data[country] is None:
        data[country] = df
    else:
        data[country] = pd.concat([data[country], df], axis=1)

data = {key: df[sorted(df.columns)] for key, df in data.items()}

for country, df in data.items():
    df.to_csv(SAVE_PATH + '/{}.csv'.format(country), encoding='utf-8')
