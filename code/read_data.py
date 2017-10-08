import numpy as np
import csv
import matplotlib.pyplot as plt

with open('UIUC Dataset/listings.csv', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    accommodates = []
    availability = []
    scores = []
    features = []
    i = 0
    scores_names = ['checkin', 'cleanliness', 'communication',
                    'location', 'rating', 'value']
    features_names = ['accomodates', 'availibility_30', 'bathrooms',
                      'bedrooms', 'beds', 'price']
    for row in reader:
        score = []
        feature = []
        # accommodates.append(float(row['accommodates']))
        # availability.append(row['availability_30'])
        if row['review_scores_checkin'] and \
                row['review_scores_cleanliness'] and \
                row['review_scores_communication'] and \
                row['review_scores_location'] and \
                row['review_scores_rating'] and row['review_scores_value']:
            if row['accommodates'] and row['availability_30'] \
                    and row['bathrooms'] and row['bedrooms'] and \
                    row['beds'] and row['price']:
                score.append(float(row['review_scores_checkin']))
                score.append(float(row['review_scores_cleanliness']))
                score.append(float(row['review_scores_communication']))
                score.append(float(row['review_scores_location']))
                score.append(float(row['review_scores_rating']))
                score.append(float(row['review_scores_value']))
                scores.append(score)
                feature.append(float(row['accommodates']))
                feature.append(float(row['availability_30']))
                feature.append(float(row['bathrooms']))
                feature.append(float(row['bedrooms']))
                feature.append(float(row['beds']))
                feature.append(float(row['price'].replace('$', "").replace(',', "")) /
                               float(row['accommodates']))
                features.append(feature)
    scores_array = np.array(scores)
    features_array = np.array(features)
    print(scores_array.shape)
    print('i = {}'.format(i))
    print(len(accommodates))
    print(len(availability))

    x = scores_array[:, 0]
    num_bins = 15
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.gca().set_yscale("log")
    # plt.show()
    for i in range(len(scores_names)):
        for j in range(len(features_names)):
            print('{}'.format(scores_names[i]), end=' vs ')
            print('{}'.format(features_names[j]), end=', ')
            print('score = {}'.format(np.corrcoef(scores_array[:, i], features_array[:, j])[0][1]))

    scores_rating = scores_array[:, 4]
    prices = features[:, -1]
    scores_keep = scores_rating >= 96 | scores_rating <= 89
    scores_kept = scores_rating[scores_keep]
    scores_kept = (scores_kept >= 96) * 1 + (scores_kept <= 89) * 1
