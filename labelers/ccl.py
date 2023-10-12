from abc import ABC, abstractmethod

import numpy as np

def union(parent, j, k):
    while parent[j] != j:
        j = parent[j]
    while parent[k] != k:
        k = parent[k]
    if k != j:
        parent[k] = j
    return parent

def find(parent, j):
    while parent[j] != j:
        j = parent[j]
    return j

def get_labels(label_img, i, j):
    if i == 0:
        labels = label_img[i, j - 1:j]
    elif j == 0:
        labels = label_img[i - 1:i + 1, j].flatten()[:-1]
    else:
        labels = label_img[i - 1:i + 1, j - 1:j + 2].flatten()[:-2]
    return labels[labels > 0]

def label_components(binary_image):
    parent = [int(0)]
    new_label = 1
    label_image = np.zeros(binary_image.shape, dtype=int)
    max_rows, max_cols = binary_image.shape
    for i in range(max_rows):
        for j in range(max_cols):
            if binary_image[i, j] == 1:
                labels = get_labels(label_image, i, j)
                if len(labels) == 0:
                    m = new_label
                    parent.append(m)
                    new_label = new_label + 1
                else:
                    m = np.min(labels)
                label_image[i, j] = m
                for label in labels:
                    if label != m:
                        parent = union(parent, m, label)
    for label in np.unique(label_image):
        if label != 0:
            label_image[label_image == label] = find(parent, label)
    for i, l in enumerate(np.unique(label_image)):
        label_image[label_image == l] = i
    return label_image
