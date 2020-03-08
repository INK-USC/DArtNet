import os
from datetime import date, timedelta
from pprint import pprint
import pickle
import numpy as np
from math import floor
from calendar import monthrange

GDELT_PATH = '/home/sankalp/tkg/data/gdelt'
EXCHANGE_PATH = '/home/sankalp/tkg/data/gdelt/Exchange_Rate_Report_all.tsv'
SAVE_PATH = '/home/sankalp/tkg/data/gdelt'

with open(f'{GDELT_PATH}/dict.pkl', 'rb') as f:
    gdelt_dict = list(pickle.load(f))

entity2num = gdelt_dict[0]
entity2country = gdelt_dict[2]

with open(EXCHANGE_PATH, 'r') as f:
    exchange_data = [a.strip().split('\t') for a in f.readlines()]

country_time_exchange = {}
time_country_exchange = {}

time_ranges = set()
for i in range(1, len(exchange_data)):
    date = exchange_data[i][0].strip().split('-')

    if date[1].lower() == 'jan':
        date = date[2] + '01' + date[0]
    elif date[1].lower() == 'feb':
        date = date[2] + '02' + date[0]
    if date[1].lower() == 'mar':
        date = date[2] + '03' + date[0]
    time_ranges.add(date)
    for j in range(1, len(exchange_data[i])):
        country = exchange_data[0][j]
        if country == 'Euro(EUR)':
            country = 'EUR'
        country_index = entity2num[country]
        if exchange_data[i][j].strip() == '':
            continue
        value = float(exchange_data[i][j])
        if country_index not in country_time_exchange:
            country_time_exchange[country_index] = {date: value}
        else:
            country_time_exchange[country_index][date] = value

        if date not in time_country_exchange:
            time_country_exchange[date] = {country_index: value}
        else:
            time_country_exchange[date][country_index] = value
        print(f'{country} {country_index} {value}')

time_ranges = list(time_ranges)
for i, t in enumerate(time_ranges):
    for c, s in country_time_exchange.items():
        if t not in s.keys():
            for j in range(i - 1, -1, -1):
                if time_ranges[j] in s.keys():
                    country_time_exchange[c][t] = country_time_exchange[c][
                        time_ranges[j]]
                    time_country_exchange[t][c] = country_time_exchange[c][
                        time_ranges[j]]
                    break

excluded_countries = []

for country, series in country_time_exchange.items():
    for t in time_ranges:
        if t not in series:
            excluded_countries.append(country)
            break

final_countries = [
    a for a in country_time_exchange.keys() if a not in excluded_countries
]

print(final_countries)
print(country_time_exchange.keys())

entities = set()
relations = set()

with open(GDELT_PATH + '/train.txt', 'r') as f:
    train_data = [
        list(map(int,
                 a.strip().split('\t')[:-1])) for a in f.readlines()
    ]
    for a in train_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])

with open(GDELT_PATH + '/valid.txt', 'r') as f:
    valid_data = [
        list(map(int,
                 a.strip().split('\t')[:-1])) for a in f.readlines()
    ]
    for a in valid_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])
with open(GDELT_PATH + '/test.txt', 'r') as f:
    test_data = [
        list(map(int,
                 a.strip().split('\t')[:-1])) for a in f.readlines()
    ]
    for a in test_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])

entities = np.array(list(entities))
relations = np.array(list(relations))
print(np.all(entities == np.arange(len(entities))))
print(np.all(relations == np.arange(len(relations))))
print(entities[0])
print(relations[0])
print(len(entities))
print(len(relations))

num_entities = len(entities)
num_relations = len(relations)
num_entities_with_time_series = len(final_countries)
print(num_entities_with_time_series)


def save_final_data(data, filename, filename2, path):
    final_data = set()
    only_att_data = set()
    only_pred_data = set()

    for head, rel, tail, time in data:
        flag_head = 0
        time = str(time)
        if time not in time_ranges:
            continue
        if head in final_countries:
            head_val = country_time_exchange[head][time]
            flag_head = 1
        else:
            head_val = 0
        flag_tail = 0
        if tail in final_countries:
            tail_val = country_time_exchange[tail][time]
            flag_tail = 1
        else:
            tail_val = 0

        final_data.add((head, rel, tail, head_val, tail_val, time, flag_head))
        final_data.add((tail, rel + num_relations, head, tail_val, head_val,
                        time, flag_tail))
        if flag_head == 1:
            only_att_data.add((head, rel, tail, head_val, tail_val, time))
        else:
            only_pred_data.add((head, rel, tail, head_val, tail_val, time))
        if flag_tail == 1:
            only_att_data.add(
                (tail, rel + num_relations, head, tail_val, head_val, time))
        else:
            only_pred_data.add(
                (tail, rel + num_relations, head, tail_val, head_val, time))

    final_data = sorted(list(final_data), key=lambda x: x[-1])
    only_att_data = sorted(list(only_att_data), key=lambda x: x[-1])
    only_pred_data = sorted(list(only_pred_data), key=lambda x: x[-1])

    with open(f'{path}/{filename}.txt', 'w') as f:
        for a in final_data:
            f.write(
                f'{a[0]}\t{a[1]}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\t{a[6]}\n')

    with open(f'{path}/{filename}_only_att_data.txt', 'w') as f:
        for a in only_att_data:
            f.write(f'{a[0]}\t{a[1]}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\n')

    with open(f'{path}/{filename2}1.txt', 'w') as f:
        for a in only_att_data:
            f.write(f'{a[0]}\t{a[1]}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\n')

    with open(f'{path}/{filename2}2.txt', 'w') as f:
        for a in only_pred_data:
            f.write(f'{a[0]}\t{a[1]}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\n')


save_final_data(train_data, 'train_processed', 'train', SAVE_PATH)
save_final_data(valid_data, 'valid_processed', 'valid', SAVE_PATH)
save_final_data(test_data, 'test_processed', 'test', SAVE_PATH)

with open(f'{SAVE_PATH}/stat.txt', 'w') as f:
    f.write(f'{num_entities}\t{int(2*num_relations)}')
