from collections import Counter
import numpy as np

def entropy(y):
    y = Counter(y) # create dictionary of all the distinct y values as key: value as the total number of it

    total = len(y)

    total_entropy = 0

    for y, num_items in y.items():
        p = num_items/total
        entropy = p*np.log(p)
        total_entropy += entropy

    return -total_entropy


def information_gain(y, y_left, y_right):
    return entropy(y) - (entropy(y_left)*len(y_left) + entropy(y_right)*len(y_right))/len(y)
