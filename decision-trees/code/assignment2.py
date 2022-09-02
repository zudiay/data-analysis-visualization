import math
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree


# Data represents each row in the given 2D dataset
class Data:
    def __init__(self, x: float, y: float, group: int):
        self.x = x
        self.y = y
        self.group = group


# Split represents decision tree splits, split groups and their entropy values
class Split:
    def __init__(self, axis: str, point: float, information_gain: float = 0, lower_group: List[Data] = None,
                 upper_group: List[Data] = None, lower_group_entropy: float = 0, upper_group_entropy: float = 0):
        self.axis = axis  # the axis of the split
        self.point = point  # the point of the split
        self.information_gain = information_gain  # the information gain the split provides
        self.lower_group = lower_group or []  # the group to the left (or bottom) side of the split
        self.upper_group = upper_group or []  # the group to the right (or top) side of the split
        self.lower_group_entropy = lower_group_entropy  # entropy of the lower group
        self.upper_group_entropy = upper_group_entropy  # entropy of the upper group


# calculates information gain given a dataset and a separation of two groups
def calculate_information_gain(dataset: List[Data], dataset_1: List[Data], dataset_2: List[Data]) -> float:
    total_entropy = calculate_entropy(dataset)
    dataset_1_entropy = calculate_entropy(dataset_1)
    dataset_2_entropy = calculate_entropy(dataset_2)
    dataset_1_weighted = (len(dataset_1) / len(dataset)) * dataset_1_entropy
    dataset_2_weighted = (len(dataset_2) / len(dataset)) * dataset_2_entropy
    information_gain = total_entropy - dataset_1_weighted - dataset_2_weighted
    return information_gain


# calculates entropy given a dataset
def calculate_entropy(dataset: List[Data]) -> float:
    group_1, group_2 = group_dataset(dataset)
    if len(group_1) == 0 or len(group_2) == 0:
        return 0  # if the dataset contains points belonging to only one group, the entropy is zero
    p1 = len(group_1) / (len(group_1) + len(group_2))
    p2 = len(group_2) / (len(group_1) + len(group_2))
    return -p1 * math.log2(p1) - p2 * math.log2(p2)


# given a dataset, separates them to two list of tuples (x,y), with respect to their groups
def group_dataset(dataset: List[Data]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    group_1, group_2 = [], []
    for data in dataset:
        if data.group == 1:
            group_1.append((data.x, data.y))
        else:
            group_2.append((data.x, data.y))
    return group_1, group_2


# given a dataset, plots them on 2D coordinate axis with different colors
def plot(dataset: List[Data], title: str):
    dataset_1, dataset_2 = group_dataset(dataset)
    dataset_1, dataset_2 = np.array(dataset_1), np.array(dataset_2)

    plt.scatter(
        dataset_1[:, 0], dataset_1[:, 1],
        c='lightblue', edgecolor='black',
        label=f'Dataset 1'
    )
    plt.scatter(
        dataset_2[:, 0], dataset_2[:, 1],
        c='yellowgreen', edgecolor='black',
        label=f'Dataset 2'
    )
    plt.title(title)


# given a dataset and a split point and axis, divides the dataset to two groups, lower and upper
def divide_dataset(dataset: List[Data], axis: str, point: float) -> Tuple[List[Data], List[Data]]:
    lower, upper = [], []
    for data in dataset:
        compare_value = data.x if axis == 'x' else data.y  # compate the value with respect to the axis
        if compare_value < point:
            lower.append(data)
        else:
            upper.append(data)
    return lower, upper


# given a dataset and an axis, tries to find the split wrt that axis with the highest information gain
def find_split(dataset: List[Data], axis: str, min_x: float, max_x: float, min_y: float, max_y: float) -> Split:
    if axis == 'x':
        min_val, max_val = min_x, max_x
    else:
        min_val, max_val = min_y, max_y
    split = Split(axis=axis, point=0)
    for i in np.linspace(min_val, max_val, 100):  # divide the range into 100 points, try splitting
        group_1, group_2 = divide_dataset(dataset, axis, i)  # divide the dataset to two at that point
        information_gain = calculate_information_gain(dataset, group_1, group_2)
        # if the information gain at that point is a better value, this will be the optimal split
        if information_gain > split.information_gain:
            split.information_gain = information_gain
            split.point = i
            split.lower_group, split.upper_group = group_1, group_2
            entropy_1, entropy_2 = calculate_entropy(group_1), calculate_entropy(group_2)
            split.lower_group_entropy, split.upper_group_entropy = entropy_1, entropy_2
    return split


# reports entropy values as the result of a split
def report_values(rank: str, split: Split):
    print(f'{rank} split is made with respect to axis {split.axis} at point {split.point:.5f}')
    if split.axis == 'x':
        print(f'Entropy left: {split.lower_group_entropy:.5f}\nEntropy right: {split.upper_group_entropy:.5f}')
    else:
        print(f'Entropy bottom: {split.lower_group_entropy:.5f}\nEntropy top: {split.upper_group_entropy:.5f}')
    weighted = (split.lower_group_entropy * len(split.lower_group) + split.upper_group_entropy * len(
        split.upper_group)) / (len(split.lower_group) + len(split.upper_group))
    print(f'Weighted average entropy: {weighted:.5f}')


# given a dataset, finds two splits using scikit-learn decision tree classifier
def calculate_scikit(dataset: List[Data]) -> Tuple[Split, Split]:
    values, target = [], []
    for data in dataset:  # create feature and target lists
        values.append([data.x, data.y])
        target.append(data.group)
    model = tree.DecisionTreeClassifier(criterion="gini", max_depth=2)
    model.fit(np.array(values), np.array(target))  # train the model with given data using decision tree classifier
    tree_text = tree.export_text(model)  # export the decision tree formed by the model as a text
    # from the decision tree text, extract the 2 split axis and points and return them as Split objects
    lines = tree_text.splitlines()
    first_split_axis = 'x' if 'feature_0' in lines[0] else 'y'
    first_split_point = float(lines[0][18:])
    second_split_axis = 'x' if 'feature_0' in lines[1] else 'y'
    second_split_point = float(lines[1][22:])
    return Split(axis=first_split_axis, point=first_split_point), Split(axis=second_split_axis, point=second_split_point)


# given a dataset and two splits, plots the split lines to visualize the decision tree
def plot_results(dataset: List[Data], first_split: Split, second_split: Split, max_x: float, max_y: float,
                 title: str, is_lower_part: bool):
    plot(dataset, title)
    second_split_min, second_split_max = 0, 1
    if second_split.axis == 'x':
        if is_lower_part:
            second_split_max = first_split.point / max_y
        else:
            second_split_min = first_split.point / max_y
        plt.axhline(y=first_split.point, color='black', linewidth=4, label=f'Boundary 1 y={first_split.point:.2f}')
        plt.axvline(x=second_split.point, ymin=second_split_min, ymax=second_split_max, color='black', linewidth=2,
                    label=f'Boundary 2 x={second_split.point:.2f}')
    else:
        if is_lower_part:
            second_split_max = first_split.point / max_x
        else:
            second_split_min = first_split.point / max_x
        plt.axvline(x=first_split.point, color='black', linewidth=4, label=f'Boundary 1 x={first_split.point:.2f}')
        plt.axhline(y=second_split.point, xmin=second_split_min, xmax=second_split_max, color='black', linewidth=2,
                    label=f'Boundary 2 y={second_split.point:.2f}')
    plt.legend()
    plt.show()


# reads the file and transforms it into a list of Data, plots it, returns the list
def read_data(file_name: str) -> List[Data]:
    dataset = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            x, y, group = line.strip().split(',')
            dataset.append(Data(x=float(x), y=float(y), group=int(group)))
    plot(dataset, 'Given Dataset')
    plt.legend()
    plt.show()
    return dataset


def decision_tree(file_name: str):
    dataset = read_data(file_name)

    # find minimum and maximum x and y values to be able to split the ranges
    min_x, max_x = min(dataset, key=lambda data: data.x).x, max(dataset, key=lambda data: data.x).x
    min_y, max_y = min(dataset, key=lambda data: data.y).y, max(dataset, key=lambda data: data.y).y

    # try splitting in y and x directions and get the best splits in both directions, select the better split
    split_x = find_split(dataset, 'x', min_x, max_x, min_y, max_y)
    split_y = find_split(dataset, 'y', min_x, max_x, min_y, max_y)
    first_split = split_x if split_x.information_gain > split_y.information_gain else split_y
    report_values('First', first_split)

    # decide which of one of the two groups to split further (the one with higher entropy)
    is_lower_part = first_split.lower_group_entropy > first_split.upper_group_entropy
    group_to_split = first_split.lower_group if is_lower_part else first_split.upper_group
    split_x = find_split(group_to_split, 'x', min_x, max_x, min_y, max_y)
    split_y = find_split(group_to_split, 'y', min_x, max_x, min_y, max_y)
    second_split = split_x if split_x.information_gain > split_y.information_gain else split_y
    report_values('Second', second_split)

    # two splits are completed, plot them
    plot_results(dataset, first_split, second_split, max_x, max_y, 'Final Classification', is_lower_part)

    # find the splits scikit-learn library forms, plot them
    scikit_split_1, scikit_split_2 = calculate_scikit(dataset)
    plot_results(dataset, scikit_split_1, scikit_split_2, max_x, max_y, 'Scikit-Learn Classification', is_lower_part)


if __name__ == '__main__':
    decision_tree('data.txt')
