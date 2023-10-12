from abc import ABC, abstractmethod

import numpy as np


class ConnectedComponentLabeler(ABC):
    labeler_type = None

    @classmethod
    def get_labeler(cls, labeler_type: str):
        return next(x for x in cls.__subclasses__() if x.labeler_type == labeler_type)()

    @abstractmethod
    def label_components(self, B):
        pass


class RecursiveConnectedComponentLabeler(ConnectedComponentLabeler):
    labeler_type = "recursive"

    def find_components(self, label_img, label):
        max_rows, max_cols = label_img.shape
        for i in range(max_rows):
            for j in range(max_cols):
                if label_img[i, j] == -1:
                    label = label + 1
                    self.search(label_img, label, i, j)

    def search(self, label_img, label, i, j):
        label_img[i, j] = label
        neighborhood = label_img[i - 1:i + 2, j - 1:j + 2]
        for n in range(neighborhood.shape[0]):
            for m in range(neighborhood.shape[1]):
                if neighborhood[n, m] == -1:
                    self.search(label_img, label, i + n - 1, j + m - 1)

    def label_components(self, binary_img):
        label_img = np.pad(-binary_img, 1, mode='constant')        
        label = 0
        self.find_components(label_img, label)
        return label_img[1:-1, 1:-1]


def get_labels(label_img, i, j):
    if i == 0:
        labels = label_img[i, j - 1:j]
    elif j == 0:
        labels = label_img[i - 1:i + 1, j].flatten()[:-1]
    else:
        labels = label_img[i - 1:i + 1, j - 1:j + 2].flatten()[:-2]
    return labels[labels > 0]


class UnionFindConnectedComponentLabeler(ConnectedComponentLabeler):
    labeler_type = "union"

    def __init__(self):
        self.parent = np.zeros(100, dtype=int)

    def union(self, j, k):
        while self.parent[j] != 0:
            j = self.parent[j]
        while self.parent[k] != 0:
            k = self.parent[k]
        if k != j:
            self.parent[k] = j

    def find(self, j):
        while self.parent[j] != 0:
            j = self.parent[j]
        return j

    def label_components(self, binary_image):
        new_label = 1
        label_image = np.zeros(binary_image.shape, dtype=int)
        max_rows, max_cols = binary_image.shape
        for i in range(max_rows):
            for j in range(max_cols):
                if binary_image[i, j] == 1:
                    labels = get_labels(label_image, i, j)
                    if len(labels) == 0:
                        m = new_label
                        new_label = new_label + 1
                    else:
                        m = np.min(labels)
                    label_image[i, j] = m
                    for label in labels:
                        if label != m:
                            self.union(m, label)
        for i in range(max_rows):
            for j in range(max_cols):
                if binary_image[i, j] == 1:
                    label_image[i, j] = self.find(label_image[i, j])
        for i, l in enumerate(np.unique(label_image)):
            label_image[label_image == l] = i
        return label_image
