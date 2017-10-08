import csv
from collections import defaultdict

MIN_SUP = 1000
POSITIVE_SCORE_THRESHOLD = 96
NEGATIVE_SCORE_THRESHOLD = 89


def normalize(s):
    return s.lower()


entity2freq = defaultdict(int)

room_types = set()
cities = set()
data = []
with open('UIUC Dataset/listings.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    line_cnt = 0
    full_score = 0
    for row in reader:
        if row['review_scores_rating'] == '':
            continue
        line_cnt += 1
        price = float(row['price'].strip('$').replace(',', ''))
        if row['accommodates'] == '':
            people = 1
        else:
            people = float(row['accommodates'])

        if row['room_type'] != '':
            room_types.add(row['room_type'])
        if row['city'] != '':
            cities.add(row['city'])

        data.append((price / people, row['room_type'],
                     cities.add(row['city'])))

print(room_types)
print(len(cities))
print(len(data))
