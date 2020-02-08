import cv2
import numpy as np

from common.models import BoundingBox


def draw_bounding_box(image_data, bounding_box, label=None, score=None, text_size=10,
                      font=cv2.FONT_HERSHEY_PLAIN, font_size=1, color=(0, 0, 0), thickness=None, info=False):
    if not isinstance(image_data, np.ndarray):
        raise ValueError('"image_data": parameter should be a numpy array.')
    if not isinstance(bounding_box, BoundingBox):
        raise ValueError('"bounding_box": parameter should be an instance of BoundingBox.')
    if not (isinstance(label, str) or (label is None)):
        raise ValueError('"label": parameter should be a string or None value.')
    if not (isinstance(score, float) or (score is None)):
        raise ValueError('"score": parameter should be a float or None value.')
    if not (isinstance(text_size, int) or isinstance(text_size, float)):
        raise ValueError('"text_size": parameter should be an integer or a float value.')
    if not (isinstance(font_size, int) or isinstance(font_size, float)):
        raise ValueError('"font_size": parameter should be an integer or a float value.')
    if not isinstance(color, tuple):
        raise ValueError('"color": parameter should be a tuple.')
    if not (isinstance(thickness, int) or (thickness is None)):
        raise ValueError('"thickness": parameter should be a float or None value.')
    if not isinstance(info, bool):
        raise ValueError('"info": parameter should be a boolean value.')
    image_height, _, _ = image_data.shape
    point_left_top = int(bounding_box.x_min), int(bounding_box.y_min)
    point_right_bottom = int(bounding_box.x_max), int(bounding_box.y_max)
    processed_image_data = image_data.copy()
    if (label is not None) or (score is not None):
        point_text = int(bounding_box.x_min), int(bounding_box.y_max - text_size)
        if 0 < (bounding_box.y_min - text_size) < image_height:
            point_text = int(bounding_box.x_min), int(bounding_box.y_min - text_size)
        elif 0 < (bounding_box.y_max + text_size) < image_height:
            point_text = int(bounding_box.x_min), int(bounding_box.y_max + text_size)
        text = ''
        if (label is not None) and (score is None):
            text = '{label}'.format(label=label)
        elif (label is not None) and (score is not None):
            text = '{label}: %{score:4.2f}'.format(label=label, score=(score * 100))
        elif (label is None) and (score is not None):
            text = '%{score:4.2f}'.format(score=(score * 100))
        if thickness is None:
            cv2.putText(img=processed_image_data, text=text, org=point_text,
                        fontFace=font, fontScale=font_size, color=color)
        else:
            cv2.putText(img=processed_image_data, text=text, org=point_text,
                        fontFace=font, fontScale=font_size, color=color, thickness=thickness)
    if thickness is None:
        cv2.rectangle(img=processed_image_data, pt1=point_left_top, pt2=point_right_bottom, color=color)
    else:
        cv2.rectangle(img=processed_image_data, pt1=point_left_top, pt2=point_right_bottom, color=color,
                      thickness=thickness)
    if info:
        print(bounding_box, 'is drawn.')
    return processed_image_data
