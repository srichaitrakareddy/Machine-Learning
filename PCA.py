#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:16:01 2019

@author: srichaitrakareddy
"""

# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.
import math
import numpy as np
import os
import graphviz
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import tree
import random
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    dictionary = {}

    for i in range(0, len(x)):
        if x[i] in dictionary.keys():
            dictionary[x[i]].append(i)
        else:
            dictionary[x[i]] = []
            dictionary[x[i]].append(i)

    return dictionary



def entropy(y, w=None):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    
    
    entropy = 0
    Y = partition(y)
    y = np.array(y)
    for i in set(y):
        if w is None:
            P = (y == i).sum()/len(y)
        else:
            P = 0
            for index in Y[i]:
                P = P + w[index]             
        entropy = entropy + P * math.log2(P)
    return -entropy


def mutual_information(x, y, w=None):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    Hy = entropy(y)
    Hy_x = 0
    X = partition(x)
    for v in X:
        if w is None
        y_temp = []
        for i in X[v]:
            y_temp.append(y[i])
        P = x.count(v)/len(x)
        Hy_x = Hy_x + P * entropy(y_temp)
        else:
            y_temp = []
            w_sum = 0
            new_w = []
            for i in X[v]:
                y_temp.append(y[i])
                new_w.append(w[i])
                w_sum = w_sum + w[i]
            P = w_sum
            Hy_x = Hy_x + P * entropy(y_temp, new_w)
    return Hy - Hy_x


def transform(y):
    y_transformed = []
    if len(set(y)) == 2:
        oneV = max(set(y))
        for value in y:
            if value == oneV:
                y_transformed.append(1)
            else:
                y_transformed.append(0)
    return y_transformed


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    if attribute_value_pairs == None or attribute_value_pairs.__len__ == 0:
        attribute_value_pairs = []    
        # Making attribute value pairs for the given training data
        for i in range(0, x.shape[1]):
            for v in set(x[:, i]):
                attribute_value_pairs.append((i,v))    

    # Base cases for the Decision Tree
    if attribute_value_pairs is None or depth == max_depth :
        y_array = np.array(y)
        freq = np.bincount(y_array)
        return np.argmax(freq)
    elif all(t == y[0] for t in y) :
        return y[0]
    else :
        #calculating the entropy for attribute pairs and choosing the best one
        
        y_transformed = transform(y)
        maxIG = 0
        for term in attribute_value_pairs:
            x_temp = []
            i = term[0]
            for j in range(0, len(x)):
                value = x[j][i]
                if value == term[1]:
                    x_temp.append(1)
                else:
                    x_temp.append(0)
            IG = mutual_information(x_temp, y_transformed, w)
            if IG >= maxIG:
                maxIG = IG
                bestsplit = term
        
        # Storing the indices of the best split and its value
        value = bestsplit[1]
        i = bestsplit[0]
        x_temp = []
        for j in range(0, len(x)):
            x_temp.append(x[j][i])
        X = partition(x_temp)
        if len(X[value]) == 0:
            for term in attribute_value_pairs:
                x_temp = []
                i = term[0]
                for j in range(0, len(x)):
                    value = x[j][i]
                    if value == term[1]:
                        x_temp.append(1)
                    else:
                        x_temp.append(0)
                IG = mutual_information(x_temp,y_transformed, w)
                if IG >= maxIG:
                    maxIG = IG
                    bestsplit = term
            value = bestsplit[1]
            i = bestsplit[0]
            x_temp = []
            for j in range(0, len(x)):
                x_temp.append(x[j][i])
            X = partition(x_temp)
        
        bestlist = X[value]
        
        trueX = []
        falseX = []
        trueY = []
        falseY = []
        
        # Separating the data into two parts based on the split
        for i in range(0, len(x)):
            temp = np.asarray(x[i])
            if i in bestlist:
                trueX.append(temp)
                trueY.append(y[i])
            else:
                falseX.append(temp)
                falseY.append(y[i])
        
        trueAVP = attribute_value_pairs.copy()
        falseAVP = attribute_value_pairs.copy()
        
        # Removing the bestsplit from attribute value pairs so that we dont use it again 
        trueAVP.remove(bestsplit)
        falseAVP.remove(bestsplit)
        
        if w is None:
            tree = {(bestsplit[0], bestsplit[1], True): id3(trueX, trueY, trueAVP, depth+1, max_depth), (bestsplit[0], bestsplit[1], False): id3(falseX, falseY, falseAVP, depth+1, max_depth)}
        else:
            #splitting the weights also into two sets depending on the best split indices
            new_tw = []
            new_fw = []
            for i in range(0, len(x)): 
                if i in bestlist:
                    new_tw.append(w[i])
                else:
                    new_fw.append(w[i])
            
            tree = {(bestsplit[0], bestsplit[1], True): id3(trueX, trueY, trueAVP, depth+1, max_depth, new_tw), (bestsplit[0], bestsplit[1], False): id3(falseX, falseY, falseAVP, depth+1, max_depth, new_fw)}
         
        return tree

#Defining bagging to implenent the bagging function.
def bagging(x, y, max_depth, num_trees)
    h_ens = []
    alpha = 1
    for k in range(1, num_trees+1):
        #get randomly generated indices
        sampleindexes = subsample(len(y))
        Xsample = x[sampleindexes].astype(int)
        ysample = []
        for index in sampleindexes:
            ysample.append(y[index])
        decision_tree = id3(Xsample, ysample, max_depth=max_depth)
        h_ens.append((alpha, decision_tree))
    return h_ens

#Defining boosting to implenent the boosting function.
def boosting(x, y, max depth, num stumps) 
    h_ens = []
    weights = np.ones(y.shape)
    weights = weights / x.shape[0]
    for i in range(1, num_stumps+1):
        model = id3(x, y, max_depth=max_depth, w=weights)
        y_pred = [predict_example(row, model) for row in x]
        weighted_error = np.dot(np.absolute(y - y_pred), weights)
        weighted_alpha = 0.5 * np.log(((1 - weighted_error) / weighted_error))
        errors = np.absolute(y - y_pred)
        for index, weight in enumerate(weights):
            if errors[index] != 0 :
                weights[index] = weights[index] * np.exp(weighted_alpha)
            else:
                weights[index] = weights[index] * np.exp(-weighted_alpha)
        weights = weights / (2 * np.sqrt(weighted_error * (1 - weighted_error)))
        h_ens.append((weighted_alpha, model))
    return h_ens

def predict_example(x, h ens),
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    try:
        l = len(tree.keys())
        
    except Exception as e:
        return tree
    
    term = list(tree.keys())[0]
    
    # Traversing the tree based on the test value
    if x[term[0]] == term[1]:
        return predict_example(x, tree[(term[0], term[1], True)])
    else:
        return predict_example(x, tree[(term[0], term[1], False)])


def compute_error(y_true, y_pred, w=None):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    count = 0
    for i in range(0, len(y_true)):
        if y_true[i] != y_pred[i]:
            count = count + 1
    return count/len(y_true)


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
     M = np.genfromtxt('./mushroom/mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]
    
    M = np.genfromtxt('./mushroom/mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    
    print("MUSHROOM DATASET: Confusion matrix (Bagging Implementation)")
    for depth in [3, 5]:
        for k in [5, 10]:
            models = bagging(Xtrn, ytrn, max_depth=depth, num_trees=k)
            y_pred = [predict_example_ens(x, models) for x in Xtst]
            print("Depth = ",depth," and Bag Size = ",k)
            print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                               columns=['Predicted Positives', 'Predicted Negatives'],
                               index=['True Positives', 'True Negatives']
                               ))
            
    print("MUSHROOM DATASET: Confusion matrix (Boosting Implementation)")
    for depth in [1, 2]:
        for stump in [5, 10]:
            models = boosting(Xtrn, ytrn, max_depth=depth, num_stumps=stump)
            y_pred = [predict_example_ens(x, models) for x in Xtst]
            print("Depth = ",depth," and Ensemble Size = ",stump)
            print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                               columns=['Predicted Positives', 'Predicted Negatives'],
                               index=['True Positives', 'True Negatives']
                               ))
            
    print("MUSHROOM DATASET: Confusion matrix (SKLearn Bagging)")
    for depth in [3, 5]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for k in [5, 10]:
            model = BaggingClassifier(base_estimator=dtree, n_estimators=k)
            model.fit(Xtrn, ytrn)
            y_pred = model.predict(Xtst)
            print("Depth = ",depth," and Bag Size = ",k)
            print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                               columns=['Predicted Positives', 'Predicted Negatives'],
                               index=['True Positives', 'True Negatives']
                               ))

    print("MUSHROOM DATASET: Confusion matrix (SKLearn Boosting)")
    for depth in [1, 2]:
        dtree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        for stump in [5, 10]:
            model = AdaBoostClassifier(base_estimator=dtree, n_estimators=stump)
            model.fit(Xtrn, ytrn)
            y_pred = model.predict(Xtst)
            print("Depth = ",depth," and Ensemble Size = ",stump)
            print(pd.DataFrame(confusion_matrix(ytst, y_pred), 
                               columns=['Predicted Positives', 'Predicted Negatives'],
                               index=['True Positives', 'True Negatives']
                               ))