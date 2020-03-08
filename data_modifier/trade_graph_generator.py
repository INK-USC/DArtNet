import os
from datetime import date
from pprint import pprint

import numpy as np
from math import floor
from calendar import monthrange

num_relations = 100

TRADE_PATH = '/home/sankalp/tkg/data/trade-data-cleaned'
EXCHANGE_PATH = '/home/sankalp/tkg/data/exchange-rate-cleaned/by_country'
SAVE_PATH = '/home/sankalp/tkg/data/trade-data-graph'

trade_files = os.listdir(TRADE_PATH)
trade_files.sort()

exchange_files = os.listdir(EXCHANGE_PATH)
exchange_files.sort()

countries_from_trade = [file[:-4] for file in trade_files]
countries_from_exchange = [file[:-4] for file in exchange_files]

print(
    f'len trade {len(countries_from_trade)} exchange {len(countries_from_exchange)}'
)

country_trade_to_exchange_dict = {}
country_exchange_to_trade_dict = {}


def check_true(a):
    for b in countries_from_trade:
        if b.lower() == a.lower() or b.lower() in a.lower() or a.lower(
        ) in b.lower():

            country_trade_to_exchange_dict[b.lower()] = a
            country_exchange_to_trade_dict[a.lower()] = b

            return True
    return False


common_countries = [a for a in countries_from_exchange if check_true(a)]
uncommon_countries = [
    a for a in countries_from_exchange if a not in common_countries
]

time_ranges = {}

for country in common_countries:
    with open(f'{EXCHANGE_PATH}/{country}.csv', 'r') as f:
        temp = f.readlines()

    d = list(map(int, temp[0].split(',')[0].split('-')))
    least_time_exchange = date(d[0], d[1], d[2])
    d = list(map(int, temp[-1].split(',')[0].split('-')))
    last_time_exchange = date(d[0], d[1], d[2])

    with open(f'{TRADE_PATH}/{country_exchange_to_trade_dict[country]}.csv',
              'r') as f:
        temp = f.readlines()[0].strip().split(',')[1:]

    d = temp[0].split('in ')[-1].split('-')
    least_time_trade = date(int(d[0]), int(d[1][1:]), 28)
    d = temp[-1].split('in ')[-1].split('-')
    last_time_trade = date(int(d[0]), int(d[1][1:]), 28)

    time_ranges[country] = [
        max(least_time_exchange, least_time_trade),
        max(last_time_exchange, last_time_trade)
    ]

pprint(time_ranges)

final_countries = [
    a for a in common_countries if time_ranges[a][0] <= date(2010, 1, 28)
]

pprint(final_countries)

date1 = date(1970, 1, 1)

num_country = 0

country_dict = {}

graph_data = set([])

min_value = 100000000000
max_value = -10000000000

min_day = 1000000000

# for file in trade_files:
for file in final_countries:
    country1 = file

    if country1 not in country_dict:

        country_dict[country1] = num_country
        num_country += 1

    country1_id = country_dict[country1]

    with open(
            "{}/{}.csv".format(TRADE_PATH,
                               country_exchange_to_trade_dict[country1]),
            'r') as f:
        trade_data = f.readlines()
        trade_data = [a.strip().split(',') for a in trade_data]

    num_col = len(trade_data[0])

    for c in range(1, len(trade_data)):
        country2 = trade_data[c][0].lower().strip().replace(',', '').replace(
            ' ', '_').replace('"', '')

        for k in range(1, len(trade_data[c]) - num_col + 1):
            country2 += trade_data[c][k].lower().strip().replace(
                ',', '').replace(' ', '_').replace('"', '')

        if country2 not in country_trade_to_exchange_dict:
            continue
        country2 = country_trade_to_exchange_dict[country2]

        if country2 not in country_dict:
            country_dict[country2] = num_country
            num_country += 1

        country2_id = country_dict[country2]

        for i in range(len(trade_data[c]) - num_col + 1, len(trade_data[c])):
            y, m = tuple(trade_data[0][i + num_col - len(
                trade_data[c])].strip().split(' ')[-1].split('-'))
            y = int(y)
            m = int(m[1:])
            date2 = date(y, m, 1)
            delta = date2 - date1
            days = delta.days
            value = trade_data[c][i].strip()
            min_day = min(min_day, days)
            # try:
            if value == '':
                value = 0
            else:
                value = float(value)
            # except Exception as e:
            #     print(trade_data[c])
            #     print(value)
            #     exit(0)

            # print([country1_id, 0, country2_id, days, value])
            if value >= 0:
                graph_data.add((country1_id, value, country2_id, days,
                                days + monthrange(y, m)[1] - 1))
            else:
                value *= -1
                graph_data.add((country2_id, value, country1_id, days,
                                days + monthrange(y, m)[1] - 1))

            min_value = min(value, min_value)

            max_value = max(value, max_value)

rel_dict = {a: 0 for a in range(num_relations)}

max_value += 1

graph_data = sorted(list(graph_data), key=lambda tup: tup[3])


def write_to_file(grd, filename):

    with open('{}/{}.csv'.format(SAVE_PATH, filename), 'w') as f:
        for gd in grd:
            for day in range(gd[-2], gd[-1] + 1):
                val = floor((float(gd[1] - min_value) * num_relations) /
                            float(max_value - min_value))
                rel_dict[val] += 1
                f.write(f'{gd[0]}\t{val}\t{gd[2]}\t{day-min_day}\n')


write_to_file(graph_data[:int(len(graph_data) * .8)], 'train')
write_to_file(graph_data[int(len(graph_data) * .8):int(len(graph_data) * .9)],
              'valid')
write_to_file(graph_data[int(len(graph_data) * .9):], 'test')
pprint(rel_dict)

with open('{}/trade_graph_entities.txt'.format(SAVE_PATH), 'w') as f:
    for k, d in country_dict.items():
        f.write('{}\t{}\n'.format(k, d))

for file in exchange_files:
    if file[:-4] not in country_dict:
        continue
    with open(f'{EXCHANGE_PATH}/{file}', 'r') as f:
        data = f.readlines()

    with open(f'{SAVE_PATH}/{country_dict[file[:-4]]}.csv', 'w') as f:
        for i in range(len(data)):
            d = data[i].strip().split(',')
            y, m, day1 = tuple(map(int, d[0].split('-')))
            date2 = date(y, m, day1)
            delta = date2 - date1
            days = delta.days

            f.write(f'{days},')
            f.write(d[1])
            f.write('\n')
