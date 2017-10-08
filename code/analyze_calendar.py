import csv
from collections import defaultdict

listing_id2state = {}
with open('UIUC Dataset/listings.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        listing_id = row['id']
        state = row['state']
        listing_id2state[listing_id] = state

with open('UIUC Dataset/econ_state.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        pass


with open('UIUC Dataset/calendar.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        listing_id = row['listing_id']
        if listing_id in listing_id2state:
            state = listing_id2state[listing_id]
            year = row['date'].split('-')[0]
            month = row['date'].split('-')[1]

            if year == '2017':
                print month
