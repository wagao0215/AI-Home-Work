import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import numpy
from IPython.display import Image
from sklearn import tree
import pydotplus
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict


data_feature_num =64
data_value = []
data_label = []

surprise_rules = [{1: [1], 2: [1], 5: [2], 26: [1]},
                  {1: [1], 2: [1], 5: [2], 27: [1]},
                  {1: [1], 2: [1], 5: [2]},
                  {1: [1], 2: [1], 26: [1]},
                  {1: [1], 2: [1], 27: [1]},
                  {5: [2], 26: [1]},
                  {5: [2], 27: [1]}]
fear_rules = [{1: [1], 2: [1], 4: [2], 5: range(1, 7), 20: range(1, 7), 25: [1]},
              {1: [1], 2: [1], 4: [2], 5: range(1, 7), 25: [1]},
              {1: [1], 2: [1], 4: [2], 5: range(1, 7), 20: range(1, 7), 25: [1], 26: [1], 27:[1]},
              {1: [1], 2: [1], 4: [2], 5: range(1, 7)},
              {5: range(1, 7), 20: range(1, 7)},
              {5: range(1, 7), 20: range(1, 7), 25: [1], 26: [1], 27: [1]}]

happy_rules = [{6: [1], 12: range(1, 7)},
               {12: [4]},
               {12: [5]}]

sadness_rules = [{1: [1], 4: [1], 11: [1], 15: [3]},
                 {1: [1], 4: [1], 11: [1], 15: [3], 54: [1], 64: [1]},
                 {1: [1], 4: [1], 15: range(1, 7)},
                 {1: [1], 4: [1], 15: range(1, 7), 54: [1], 64: [1]},
                 {6: [1], 15: range(1, 7)},
                 {6: [1], 15: range(1, 7), 54: [1], 64: [1]},
                 {1: [1], 4: [1], 11: [1]},
                 {1: [1], 4: [1], 11: [1], 54: [1], 64: [1]},
                 {1: [1], 4: [1], 15: [3], 17: [1]},
                 {1: [1], 4: [1], 15: [3], 17: [1], 54: [1], 64: [1]},
                 {11: [1], 15: [3]},
                 {11: [1], 15: [3], 54: [1], 64: [1]},
                 {11: [1], 17: [1]}
                 ]

disgust_rules = [{9: [1]},
                 {9: [1], 16: [1], 15: [1]},  # check typo?
                 {9: [1], 16: [1], 25: [1]},  # check typo?
                 {9: [1], 16: [1], 26: [1]},
                 {9: [1], 17: [1]},
                 {10: range(1, 7)},
                 {10: range(1, 7), 16: [1], 25: [1]},  # check typo?
                 {10: range(1, 7), 16: [1], 26: [1]},  # check typo?
                 {10: [1], 17: [1]}]

anger_rules = [{4: [1], 5: range(1, 7), 7: [1], 10: range(1, 7), 22: [1], 23: [1], 25: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 10: range(1, 7), 22: [1], 23: [1], 26: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 10: range(1, 7), 23: [1], 25: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 10: range(1, 7), 23: [1], 26: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 23: [1], 25: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 23: [1], 26: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 17: [1], 23: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 17: [1], 24: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 23: [1]},
               {4: [1], 5: range(1, 7), 7: [1], 24: [1]}]


def data_preprocess():
    with open('Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.csv', 'rb') as csv_file:
        reader = csv.reader(csv_file)
        for readline in reader:
            new_vector = [0] * data_feature_num
            # data_label.append(int(readline[1]))
            features = readline[2].split('+')
            # print features
            for i in features:
                start_index = 0 if i[0].isdigit() else 1
                end_index = len(i) if i[-1].isdigit() else len(i) - 1
                new_vector[int(i[start_index: end_index]) - 1] = 1 if i[-1].isdigit() else ("abcde").index(i[-1]) + 2
            features_description = []
            for i in range(data_feature_num):
                if new_vector[i] != 0:
                    features_description.append(i + 1)
            # print "Data value:", readline[2], "Convert:", features_description
            # print new_vector
            # print "Data label:", readline[1], "Convert:", data_label[-1]
            data_value.append(new_vector)
        csv_file.close()


def add_label():
    potenial_label = []
    for i in range(len(data_value)):
        potenial_label.append([0] * 6)
    # print potenial_label
    count = 0
    for data_vector in data_value:
        current_rules = surprise_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][0] = 1

        current_rules = fear_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][1] = 1

        current_rules = happy_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][2] = 1

        current_rules = sadness_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][3] = 1

        current_rules = disgust_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][4] = 1

        current_rules = anger_rules
        rules_match_result = [1] * len(current_rules)
        for i in range(data_feature_num):
            for j in range(len(rules_match_result)):
                # if (not i + 1 in current_rules[j] and data_vector[i] != 0)\
                #         or (i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]):
                if i + 1 in current_rules[j] and not data_vector[i] in current_rules[j][i + 1]:
                    rules_match_result[j] = 0
        if max(rules_match_result) > 0:
            potenial_label[count][5] = 1

        count += 1
    # print potenial_label

    for i in range(len(potenial_label)):
        data_label.append(potenial_label[i].index(max(potenial_label[i])) + 1 if max(potenial_label[i]) > 0 else 0)
        if sum(potenial_label[i]) > 1:
            print i, potenial_label[i]
    print data_label


# data_preprocess()
# add_label()
# print data_label
# classifier = LinearSVC()
# classifier = classifier.fit(data_value[0:int(len(data_value)*0.8)], data_label[0:int(len(data_value)*0.8)])
# data_predict = classifier.predict(data_value[int(len(data_value)*0.8):])
#
# matrix = confusion_matrix(data_label[int(len(data_value)*0.8):], data_predict)
#
# print matrix
#
# print len(data_value) - int(len(data_value)*0.8)

with open('output_3.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    count = 0
    for readline in reader:
        print readline
        print readline[0: -1]
        print readline[-1]
        if count != 0:
            data_value.append(readline[0: -2])
            data_label.append(readline[-1])

        count += 1
        # new_vector = [0] * data_feature_num
        # # data_label.append(int(readline[1]))
        # features = readline[2].split('+')
        # # print features
        # for i in features:
        #     start_index = 0 if i[0].isdigit() else 1
        #     end_index = len(i) if i[-1].isdigit() else len(i) - 1
        #     new_vector[int(i[start_index: end_index]) - 1] = 1 if i[-1].isdigit() else ("abcde").index(i[-1]) + 2
        # features_description = []
        # for i in range(data_feature_num):
        #     if new_vector[i] != 0:
        #         features_description.append(i + 1)
        # # print "Data value:", readline[2], "Convert:", features_description
        # # print new_vector
        # # print "Data label:", readline[1], "Convert:", data_label[-1]
        # data_value.append(new_vector)
    csv_file.close()

# with open('output.csv', 'wb') as csvfile:
#     writer = csv.writer(csvfile)
#     output_line = ""
#     for i in range(data_feature_num):
#         output_line += str((i + 1) % 10)
#         print output_line
#     writer.writerow(output_line)
#     for i in range(len(data_value)):
#         output_line = ""
#         for j in range(data_feature_num):
#             output_line += str(data_value[i][j])
#         output_line += str(data_label[i])
#         print output_line
#         writer.writerow(output_line)

# classifier = SVC(kernel='rbf', decision_function_shape='ovo')
# classifier = DecisionTreeClassifier()
classifier = RandomForestClassifier(n_estimators=6)
classifier = classifier.fit(data_value[0:int(len(data_value)*0.8)], data_label[0:int(len(data_value)*0.8)])
data_predict = classifier.predict(data_value[int(len(data_value)*0.8):])

matrix = confusion_matrix(data_label[int(len(data_value)*0.8):], data_predict)

print matrix

print len(data_value) - int(len(data_value)*0.8)

count = 0
for i in range(len(data_label)):
    if data_label[i] > 0:
        count += 1

print count

print classifier.score(data_value[0:int(len(data_value)*0.8)], data_label[0:int(len(data_value)*0.8)])

predict = cross_val_predict(classifier, data_value, data_label)

print predict

print confusion_matrix(data_label, predict)

# with open("output.dot", 'w') as f:
#     f = tree.export_graphviz(classifier, out_file=f)
#
# os.unlink('output.dot')
#
# dot_data = tree.export_graphviz(classifier, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("output.pdf")
#
# dot_data = tree.export_graphviz(classifier, out_file=None,
#                          feature_names=[2] * data_feature_num,
#                          class_names=data_label,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())