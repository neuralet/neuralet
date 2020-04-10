# import the necessary packages
from collections import OrderedDict

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    """
    A simple object tracker based on euclidean distance of bounding boxes centroid of two consecutive frames.
    if a box is losted betweeb two frames the tracker keep the box for next max_disappeared frames.

    :param max_disappeared: If a box is losted betweeb two frames the tracker keep the box for next
     max_disappeared frames.
    """

    def __init__(self, max_disappeared=50):
        self.nextobject_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, object_item):
        self.objects[self.nextobject_id] = object_item
        self.disappeared[self.nextobject_id] = 0
        self.nextobject_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, object_list):
        if len(object_list) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        input_centroids = np.zeros((len(object_list), 2))
        for i, object_item in enumerate(object_list):
            input_centroids[i] = (object_item["centroid"][0], object_item["centroid"][1])
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(object_list[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [object_item["centroid"][0:2] for object_item in self.objects.values()]
            computed_dist = dist.cdist(np.array(object_centroids), input_centroids)
            rows = computed_dist.min(axis=1).argsort()
            cols = computed_dist.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = object_list[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, computed_dist.shape[0])).difference(used_rows)
            unused_cols = set(range(0, computed_dist.shape[1])).difference(used_cols)

            if computed_dist.shape[0] >= computed_dist.shape[1]:
                biggest_existing_id = int(object_list[-1]["id"].split("-")[-1])

                for row in unused_rows:
                    object_id = object_ids[row]
                    self.objects[object_id]["id"] = self.objects[object_id]["id"].split("-")[0] + "-" + str(
                        biggest_existing_id + 1)
                    self.disappeared[object_id] += 1
                    biggest_existing_id += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            else:
                for col in unused_cols:
                    self.register(object_list[col])

        return self.objects
