import os
from pprint import pprint

import numpy as np
import pandas as pd
from datetime import date

import difflib
from copy import deepcopy


def get_date_from_string(d):

    if d.lower() == 'jan':
        return 1
    if d.lower() == 'feb':
        return 2
    if d.lower() == 'mar':
        return 3
    if d.lower() == 'apr':
        return 4
    if d.lower() == 'may':
        return 5
    if d.lower() == 'jun':
        return 6
    if d.lower() == 'jul':
        return 7
    if d.lower() == 'aug':
        return 8
    if d.lower() == 'sep':
        return 9
    if d.lower() == 'oct':
        return 10
    if d.lower() == 'nov':
        return 11
    if d.lower() == 'dec':
        return 12


DATA_PATH = '/home/sankalp/tkg/data/exchange-rate'
SAVE_PATH = '/home/sankalp/tkg/data/exchange-rate-cleaned'
files = os.listdir(DATA_PATH)
files.sort()

countries = []

for file in files:
    print(file)
    with open('{}/{}'.format(DATA_PATH, file), 'r', encoding="latin-1") as f:
        text = f.read()

    text = text.split('COUNTRIES')[-1].split(
        'Disclaimer')[0].strip().lower().replace('\t', '').replace('\n', ',')

    countries += text.split(',')

countries = list(set(countries))
countries = [a.strip().replace(' ', '_') for a in countries]
countries.sort()
print(countries)
print(len(countries))

currency_dict = {}

for file in files:

    with open('{}/{}'.format(DATA_PATH, file), 'r', encoding="latin-1") as f:
        text = f.read().replace('\xa0', ' ').encode('ascii').decode('latin-1')

    text = text.split('USER')[0].strip().split('\n')[1:]

    cols = text[0].strip().split('\t')

    for t in cols[1:]:
        if t not in currency_dict:
            currency_dict[t] = []

    for te in text[1:]:
        ent = te.strip().split('\t')
        dt = ent[0].split('-')
        dt = date(int(dt[2]), get_date_from_string(dt[1]), int(dt[0]))
        for i in range(1, len(ent)):
            currency_dict[cols[i]].append((dt, ent[i].strip()))

for k in currency_dict:
    currency_dict[k].sort(key=lambda tup: tup[0])

for k, dat in currency_dict.items():
    with open('{}/by_currency/{}.csv'.format(SAVE_PATH, k), 'w') as f:
        for d in dat:
            f.write('{},{}\n'.format(d[0], d[1]))

currency_to_country = {}

for k in currency_dict:
    currency_to_country[k] = list(
        difflib.get_close_matches(k.split(' ')[0], countries, n=1, cutoff=0.2))

currency_to_country['Bolivar Fuerte(VEF)'] = ['venezuela']
currency_to_country['Bolivar Soberano(VES)'] = ['venezuela']
currency_to_country['Brazilian real(BRL)'] = ['brazil']
currency_to_country['Brunei dollar(BND)'] = ['brunei_darussalam']
currency_to_country['Canadian dollar(CAD)'] = ['canada']
currency_to_country['Chilean peso(CLP)'] = ['chile']
currency_to_country['Chinese yuan(CNY)'] = ['china']
currency_to_country['Danish krone(DKK)'] = ['denmark']
currency_to_country['Euro(EUR)'] = [
    'germany', 'france', 'italy', 'spain', 'greece', 'portugal', 'netherlands',
    'austria', 'belgium', 'ireland', 'finland', 'slovak_republic', 'estonia',
    'cyprus', 'slovenia', 'malta', 'luxembourg', 'san_marino', 'netherlands'
]
currency_to_country['Iranian rial(IRR)'] = ['iran']
currency_to_country['New Zealand dollar(NZD)'] = ['new_zealand']
currency_to_country['Norwegian krone(NOK)'] = ['norway']
currency_to_country['Peruvian sol(PEN)'] = ['peru']
currency_to_country['Polish zloty(PLN)'] = ['poland']
currency_to_country['Russian ruble(RUB)'] = ['russian_federation']
currency_to_country['Saudi Arabian riyal(SAR)'] = ['saudi_arabia']
currency_to_country['Sri Lankan rupee(LKR)'] = ['sri_lanka']
currency_to_country['Swiss franc(CHF)'] = ['switzerland']
currency_to_country['Thai baht(THB)'] = ['thailand']
currency_to_country['Swiss franc(CHF)'] = ['switzerland']
currency_to_country['Trinidadian dollar(TTD)'] = ['trinidad_and_tobago']
currency_to_country['U.A.E. dirham(AED)'] = ['united_arab_emirates']
currency_to_country['U.K. pound(GBP)'] = ['united_kingdom']
currency_to_country['U.S. dollar(USD)'] = ['united_states']

pprint(currency_to_country)

pprint(countries)

country_dict = {}

for curr, dat in currency_dict.items():

    country_list = currency_to_country[curr]
    for country in country_list:
        if country in country_dict:
            country_dict[country] += deepcopy(dat)
        else:
            country_dict[country] = deepcopy(dat)

for k in country_dict:

    country_dict[k].sort(key=lambda tup: tup[0])
    i = 0
    for i in range(len(country_dict[k])):
        if country_dict[k][i][1] != '':
            break
    country_dict[k] = country_dict[k][i:]

for k, dat in country_dict.items():
    with open('{}/by_country/{}.csv'.format(SAVE_PATH, k), 'w') as f:
        for d in dat:
            f.write('{},{}\n'.format(d[0], d[1]))