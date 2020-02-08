from common.models import BoundingBox, GroundTruth, Detection


def official_line_to_ground_truth_mapper(line, info=False):
    if not isinstance(line, str):
        raise ValueError('"line": parameter should be a string value.')
    if not isinstance(info, bool):
        raise ValueError('"info": parameter should be a boolean value.')
    words = line.split(sep=' ')
    if len(words) < 8:
        raise ValueError('"line": not official ground truth.')
    try:
        label = str(words[0])
        x_min = float(words[4])
        y_min = float(words[5])
        x_max = float(words[6])
        y_max = float(words[7])
        bounding_box = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        if info:
            print(bounding_box, 'is mapped.')
        return GroundTruth(label=label, bounding_box=bounding_box)
    except ValueError:
        raise ValueError('"line": not suitable.')


def ground_truth_to_line_mapper(ground_truth, info=False):
    if not isinstance(ground_truth, GroundTruth):
        raise ValueError('"ground_truth": parameter should be an instance of GroundTruth.')
    if not isinstance(info, bool):
        raise ValueError('"info": parameter should be a boolean value.')
    formatted_line = "{label} {x_min} {y_min} {x_max} {y_max}"
    line = formatted_line.format(label=str(ground_truth.label),
                                 x_min=str(ground_truth.bounding_box.x_min),
                                 x_max=str(ground_truth.bounding_box.x_max),
                                 y_min=str(ground_truth.bounding_box.y_min),
                                 y_max=str(ground_truth.bounding_box.y_max))
    if info:
        print(ground_truth, 'is mapped.')
    return line


def detection_to_line_mapper(detection, info=False):
    if not isinstance(detection, Detection):
        raise ValueError('"detection": parameter should be an instance of Detection.')
    if not isinstance(info, bool):
        raise ValueError('"info": parameter should be a boolean value.')
    formatted_line = "{label} {score} {x_min} {y_min} {x_max} {y_max}"
    line = formatted_line.format(label=str(detection.label),
                                 score=str(detection.score),
                                 x_min=str(detection.bounding_box.x_min),
                                 x_max=str(detection.bounding_box.x_max),
                                 y_min=str(detection.bounding_box.y_min),
                                 y_max=str(detection.bounding_box.y_max))
    if info:
        print(detection, 'is mapped.')
    return line
