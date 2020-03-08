import os
from datetime import date, timedelta
from pprint import pprint
import pickle
import numpy as np
from math import floor
from calendar import monthrange

DATA_PATH = '/home/sankalp/tkg/data/cite20k'
CITATION_PATH = '/home/sankalp/tkg/data/cite20k/citations.csv'
SAVE_PATH = '/home/sankalp/tkg/data/cite20k'

with open(CITATION_PATH, 'r') as f:
    citation_data = [
        list(map(int, list(map(float,
                               a.strip().split(','))))) for a in f.readlines()
    ]

# print(citation_data)

id_time_citation = {}
time_id_citation = {}

time_ranges = set()

for i in range(1, len(citation_data)):
    for j in range(len(citation_data[i])):
        date = citation_data[0][j]
        if date == 2019:
            continue
        time_ranges.add(date)
        cite = citation_data[i][j]
        index = i - 1
        if index not in id_time_citation:
            id_time_citation[index] = {date: cite}
        else:
            id_time_citation[index][date] = cite

        if date not in time_id_citation:
            time_id_citation[date] = {index: cite}
        else:
            time_id_citation[date][index] = cite
        # print(f' {index} {cite}')

included_times = []

for t, id_cite in time_id_citation.items():
    for _, val in id_cite.items():
        if val != 0:
            included_times.append(t)
            break

print(included_times)

print(f'{len(included_times)} {len(time_ranges)}')

time_non_zero_citation = {}

for t in included_times:
    cite = 0
    for _, val in time_id_citation[t].items():
        if val != 0:
            cite += 1
    time_non_zero_citation[t] = cite

pprint(time_non_zero_citation)

entities = set()
relations = set()

with open(DATA_PATH + '/train.txt', 'r') as f:
    train_data = [
        list(map(int,
                 a.strip().split(' ')[:-1])) for a in f.readlines()
    ]
    for a in train_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])

with open(DATA_PATH + '/valid.txt', 'r') as f:
    valid_data = [
        list(map(int,
                 a.strip().split(' ')[:-1])) for a in f.readlines()
    ]
    for a in valid_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])
with open(DATA_PATH + '/test.txt', 'r') as f:
    test_data = [
        list(map(int,
                 a.strip().split(' ')[:-1])) for a in f.readlines()
    ]
    for a in test_data:
        entities.add(a[0])
        entities.add(a[2])
        relations.add(a[1])

entities = np.array(list(entities))
relations = np.array(list(relations))
print(entities)
print(np.all(entities == np.arange(len(entities))))
print(np.all(relations == np.arange(len(relations))))
print(entities[0])
print(relations[0])
print(len(entities))
print(len(relations))

final_entity_mapping = {}

num_relations = 1

for i, j in enumerate(list(entities)):
    final_entity_mapping[j] = i
    num_entities = i + 1


def save_final_data(data, filename, path):
    final_data = set()

    for head, rel, tail, time in data:
        if time not in included_times:
            continue
        val_head = time_id_citation[time][head] / 10000
        val_tail = time_id_citation[time][tail] / 10000
        head = final_entity_mapping[head]
        tail = final_entity_mapping[tail]

        final_data.add((head, rel, tail, val_head, val_tail, time))
        # final_data.add(
        #     (tail, rel + num_relations, head, val_tail, val_head, time))

    final_data = sorted(list(final_data), key=lambda x: x[-1])

    with open(f'{path}/{filename}.txt', 'w') as f:
        for a in final_data:
            f.write(f'{a[0]}\t{a[1]}\t{a[2]}\t{a[3]}\t{a[4]}\t{a[5]}\n')


save_final_data(train_data, 'train_processed', SAVE_PATH)
save_final_data(valid_data, 'valid_processed', SAVE_PATH)
save_final_data(test_data, 'test_processed', SAVE_PATH)

with open(f'{SAVE_PATH}/stat.txt', 'w') as f:
    f.write(f'{num_entities}\t{int(num_relations)}')
