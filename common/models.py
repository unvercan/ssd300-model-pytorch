from common.helpers import convert_instance_to_text


class BoundingBox(object):
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.width = abs(x_max - x_min)
        self.height = abs(y_max - y_min)

    def __str__(self):
        return convert_instance_to_text(self)

    def __repr__(self):
        return self.__str__()


class GroundTruth(object):
    def __init__(self, label, bounding_box):
        self.label = label
        self.bounding_box = bounding_box

    def __str__(self):
        return convert_instance_to_text(self)

    def __repr__(self):
        return self.__str__()


class Detection(object):
    def __init__(self, label, score, bounding_box):
        self.label = label
        self.score = score
        self.bounding_box = bounding_box

    def __str__(self):
        return convert_instance_to_text(self)

    def __repr__(self):
        return self.__str__()
