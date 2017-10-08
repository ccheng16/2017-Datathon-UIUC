import csv
from collections import defaultdict

MIN_SUP = 1000
POSITIVE_SCORE_THRESHOLD = 96
NEGATIVE_SCORE_THRESHOLD = 89
TOP_K = 20


def normalize(s):
    return s.lower()


for room_type in ['Shared room', 'Entire home/apt', 'Private room', 'Overall']:
    print '\n\n'
    print 'Rome Type =', room_type

    entity2freq = defaultdict(int)

    data = []
    with open('UIUC Dataset/listings.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        line_cnt = 0
        full_score = 0
        for row in reader:
            if row['review_scores_rating'] == '':
                continue
            line_cnt += 1
            amenities = row['amenities'].strip('{}')
            tokens = amenities.split(',')
            entities = []
            for token in tokens:
                if len(token) == 0:
                    continue
                if token[0] == '"':
                    if token[-1] != '"':
                        print 'Error'
                entity = normalize(token.strip('"'))
                entity2freq[entity] += 1
                entities.append(entity)
            score = float(row['review_scores_rating'])
            if score >= 96:
                full_score += 1

            if row['room_type'] == room_type or room_type == 'Overall':
                data.append((score, entities))

        print('# of listings with reviews = ', line_cnt)
        print('# of full score = ', full_score,
              float(full_score) / line_cnt * 100)

    print '# of different amenities = ', len(entity2freq)
    for (entity, freq) in entity2freq.iteritems():
        if freq > MIN_SUP:
            # print entity, freq
            pass

    sorted_data = sorted(data, key=lambda x: -x[0])

    freq = [defaultdict(int), defaultdict(int)]
    sizes = [0, 0]
    for (score, entities) in sorted_data:
        label = -1
        if score >= POSITIVE_SCORE_THRESHOLD:
            label = 1
            sizes[1] += 1
        elif score <= NEGATIVE_SCORE_THRESHOLD:
            label = 0
            sizes[0] += 1
        if label != -1:
            for entity in entities:
                if entity2freq[entity] >= MIN_SUP:
                    freq[label][entity] += 1

    label = 1
    pairs = []
    for (entity, entity_freq) in entity2freq.iteritems():
        if entity_freq > MIN_SUP:
            if freq[label][entity] + freq[1 - label][entity] == 0:
                continue
            relative_freq_label = float(freq[label][entity]) / sizes[label]
            relative_freq_other =\
                float(freq[1 - label][entity]) / sizes[1 - label]
            interestingness = \
                relative_freq_label ** 2 / (relative_freq_other + 1e-8)
            confidence =\
                float(freq[label][entity]) /\
                (freq[label][entity] + freq[1 - label][entity])
            pairs.append((confidence, interestingness, entity))

    print('=============Based on Confidence=============')
    pairs = sorted(pairs, key=lambda x: -x[0])
    for (confidence, interestingness, entity) in pairs[:TOP_K]:
        relative_freq_label = float(freq[label][entity]) / sizes[label]
        relative_freq_other = float(freq[1 - label][entity]) / sizes[1 - label]
        print entity, relative_freq_label, relative_freq_other, confidence
        # print entity, confidence

    print('=============Based on Interestingness=============')
    pairs = sorted(pairs, key=lambda x: -x[1])
    for (confidence, interestingness, entity) in pairs[:TOP_K]:
        relative_freq_label = float(freq[label][entity]) / sizes[label]
        relative_freq_other = float(freq[1 - label][entity]) / sizes[1 - label]
        print(entity, relative_freq_label,
              relative_freq_other, interestingness, confidence)
        # print entity, interestingness
