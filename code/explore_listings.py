import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# read data
with open("./listings.csv", 'r') as f:
    # csv_file = csv.reader(f, delimiter='\t')
    airbnb_listings = pd.read_table(
        f, sep=',', header=0,
        lineterminator='\n', encoding="utf-8")

review_list = ['review_scores_checkin',
               'review_scores_cleanliness',
               'review_scores_communication',
               'review_scores_location',
               'review_scores_value']
overall_review = ['review_scores_rating']

# get infos about null in data frame
x = airbnb_listings[review_list + overall_review].isnull().sum(axis=1)
print(sum(x > 0))
row_index = np.where(x == 0)[0]
filetered_listings = airbnb_listings.iloc[row_index]
print(filetered_listings.shape)

# linear regression assumption on dataset
X = filetered_listings[review_list]
y = filetered_listings[overall_review]
index = range(len(X.index.values))
index_dict = defaultdict(list)
index_dict = defaultdict(list)

for i in index:
    index_dict[i % 5].append(i)


# split dataset
def split_dataset_linear(X, y):
    X_dict = defaultdict(np.array)
    y_dict = defaultdict(np.array)
    for i in range(5):
        X_dict[i] = np.array(X.iloc[index_dict[i]])
        y_dict[i] = np.array(y.iloc[index_dict[i]])
    return X_dict, y_dict


# implement linear regression
# with epoch round k-fold cross validation
def fit_model(X, y, split_method, model, epoch=2, fold=5, mydict={}):
    X_dict, y_dict = split_method(X, y)
    all_regr = []
    all_y_train = []
    all_y_test = []
    all_pred_train = []
    all_pred_test = []
    for ep in range(epoch):
        for i in range(fold):
            regr = model(**mydict)
            index = list(range(i))
            index.extend(list(range(i + 1, fold)))
            X_test = X_dict[i]
            y_test = y_dict[i]
            X_train = X_dict[index[0]]
            y_train = y_dict[index[0]]
            for j in index[1:]:
                X_train = np.concatenate([X_train, X_dict[j]], axis=0)
                y_train = np.concatenate([y_train, y_dict[j]], axis=0)
            regr.fit(X_train, y_train)
            pred_train = regr.predict(X_train)
            pred_test = regr.predict(X_test)
            # get results
            all_regr.append(regr)
            all_y_train.append(y_train)
            all_y_test.append(y_test)
            all_pred_train.append(pred_train)
            all_pred_test.append(pred_test)
    return all_regr, all_pred_train, all_pred_test, all_y_train, all_y_test


all_regr, all_pred_train, all_pred_test, all_y_train, all_y_test =\
    fit_model(X, y, split_dataset_linear, linear_model.LinearRegression)
for i in range(len(all_regr)):
    regr = all_regr[i]
    pred_train = all_pred_train[i]
    y_train = all_y_train[i]
    pred_test = all_pred_test[i]
    y_test = all_y_test[i]
    print(regr.coef_, regr.intercept_)
    print("Rank of the coefficient {}".format(np.argsort(regr.coef_[0])))
    print("For test on split index {}, "
          "train error is {}, "
          "test error is {}\n".format(i, mean_squared_error(y_train, pred_train),
                                      mean_squared_error(y_test, pred_test)))


# Change question to classification problem
# split dataset
def split_dataset_logit(X, y):
    X_dict = defaultdict(np.array)
    y_dict = defaultdict(np.array)
    for i in range(5):
        X_dict[i] = np.array(X.iloc[index_dict[i]])
        y_dict[i] = np.array(y.iloc[index_dict[i]]).ravel()
    return X_dict, y_dict


#  get labels
y_label = (y >= 96)
all_regr, all_pred_train, all_pred_test, all_y_train, all_y_test =\
    fit_model(X, y_label, split_dataset_logit,
              linear_model.LogisticRegression, mydict={'C': 1e5})

all_coef = defaultdict(list)
all_accuracy_train = []
all_accuracy_test = []
for i in range(len(all_regr)):
    regr = all_regr[i]
    pred_train = all_pred_train[i]
    y_train = all_y_train[i]
    pred_test = all_pred_test[i]
    y_test = all_y_test[i]

    accuracy_train = sum(pred_train == y_train) / len(y_train)
    accuracy_test = sum(pred_test == y_test) / len(y_test)
    print(regr.coef_, regr.intercept_)
    print("Rank of the coefficient {}".format(np.argsort(regr.coef_[0])))
    print("For test on split index {}, "
          "train accuracy is {}, "
          "test accuracy is {}\n".format(i, accuracy_train, accuracy_test))

    # store coefs and accuracy
    all_accuracy_train.append(accuracy_train)
    all_accuracy_test.append(accuracy_test)
    for j in range(len(regr.coef_[0])):
        all_coef[j].append(regr.coef_[0][j])


# calculate the vairance of accuracy and coefficients
var_acc_train = np.var(all_accuracy_train, ddof=1)
var_acc_test = np.var(all_accuracy_test, ddof=1)
var_coef = defaultdict(float)

for i in all_coef.keys():
    var_coef[i] = np.var(all_coef[i], ddof=1)
    print("{}th coefficient: "
          "mean {}, var {}".format(i, np.mean(all_coef[i]), var_coef[i]))

print("Accuracy for training: "
      "mean {}, var {}".format(np.mean(all_accuracy_train), var_acc_train))
print("Accuracy for test: "
      "mean {}, var {}".format(np.mean(all_accuracy_test), var_acc_test))
