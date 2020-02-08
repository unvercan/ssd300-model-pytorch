import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from common.file_operations import get_file_paths_in_directory, filter_file_paths_by_extensions, save_file
from common.image_operations import draw_bounding_box
from common.mappers import detection_to_line_mapper
from common.models import BoundingBox, Detection
from evaluation.voc0712 import configuration
from ssd300.ssd300 import build_ssd300


# pre-process image
def pre_process_image(image_data, channel_mean=None):
    if not isinstance(image_data, np.ndarray):
        raise ValueError('"image_data": parameter should be a numpy array')
    pre_processed_data = image_data.copy()
    if channel_mean is not None:
        pre_processed_data = pre_processed_data - channel_mean
    pre_processed_data = cv2.resize(src=pre_processed_data, dsize=(300, 300)).astype(np.float32)
    return pre_processed_data


def main():
    # paths
    path = dict()

    # pre-trained weight path
    path['pre_trained_weight'] = 'C:\\weights\\ssd_300_VOC0712.pth'

    # root paths
    path['images'] = 'C:\\images'
    path['detections'] = 'C:\\detections'

    # detection results root paths
    path['detections_text'] = os.path.join(path['detections'], 'text')
    path['detections_image'] = os.path.join(path['detections'], 'image')

    # make directories if not exist
    for key in path.keys():
        value = path[key]
        if not os.path.exists(value):
            os.makedirs(value)

    # parameters
    log = True
    cuda_enabled = True
    number_of_samples = 5
    thickness = 1

    # colors
    color = dict()
    color['red'] = (0, 0, 255)

    # thresholds
    threshold = dict()
    threshold['detection'] = 0.001
    threshold['draw'] = 0.3

    # extensions
    extension = dict()
    extension['detection'] = 'txt'
    extension['image'] = 'jpg'

    # enable cuda tensor
    if cuda_enabled and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # get image paths
    file_paths = get_file_paths_in_directory(directory=path['images'])
    image_file_paths = filter_file_paths_by_extensions(file_paths=file_paths, extensions=[extension['image']])

    # limit samples
    image_file_paths = image_file_paths[0:number_of_samples]
    assert len(image_file_paths) > 0

    # calculate images channel mean
    channel_mean = np.zeros(shape=(3,))
    for image_file_path in image_file_paths:
        # load image file
        image_data = cv2.imread(filename=image_file_path, flags=cv2.IMREAD_COLOR)
        if log:
            print(image_file_path, 'is loaded.')
        image_channel_mean = np.mean(image_data, axis=0)
        image_channel_mean = np.mean(image_channel_mean, axis=0)
        channel_mean += image_channel_mean
    number_of_training_images = len(image_file_paths)
    channel_mean /= float(number_of_training_images)
    if log:
        print("Images channel mean:", channel_mean)

    # initialize SSD300
    network = build_ssd300(configuration=configuration)

    # load pre-trained weights
    network.custom_pre_trained_loader(pre_trained_file=path['pre_trained_weight'])
    network.eval()

    # loop
    for sample_index in range(number_of_samples):
        # file paths and file names
        image_file_path = image_file_paths[sample_index]
        image_name = image_file_path.split(sep='\\')[-1].split(sep='.')[0]
        detection_file_name = image_name + '.' + extension['detection']
        image_file_name = image_name + '.' + extension['image']
        detection_file_path = os.path.join(path['detections_text'], detection_file_name)
        detection_image_file_path = os.path.join(path['detections_image'], image_file_name)

        # load image file
        image_data = cv2.imread(filename=image_file_path, flags=cv2.IMREAD_COLOR)
        image_height, image_width, _ = image_data.shape
        if log:
            print(image_file_path, 'is loaded.')

        # detect objects
        detections = []
        pre_processed = pre_process_image(image_data=image_data, channel_mean=channel_mean)
        input_tensor = torch.from_numpy(pre_processed)
        input_tensor = input_tensor.permute(dims=(2, 0, 1))
        input_tensor = Variable(input_tensor.unsqueeze(dim=0))
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        output_tensor = network(input_tensor)
        output_data = output_tensor.data.cpu().numpy()
        number_of_detected_labels = output_data.shape[1]
        number_of_detection = output_data.shape[2]
        for label_index in range(number_of_detected_labels):
            for detection_index in range(number_of_detection):
                score = float(output_data[0, label_index, detection_index, 0])
                if score > threshold['detection']:
                    label = configuration['labels'][label_index - 1]
                    x_min_relative = output_data[0, label_index, detection_index, 1]
                    x_max_relative = output_data[0, label_index, detection_index, 3]
                    y_min_relative = output_data[0, label_index, detection_index, 2]
                    y_max_relative = output_data[0, label_index, detection_index, 4]
                    x_min_absolute = x_min_relative * image_width
                    x_max_absolute = x_max_relative * image_width
                    y_min_absolute = y_min_relative * image_height
                    y_max_absolute = y_max_relative * image_height
                    bounding_box = BoundingBox(x_min=x_min_absolute, x_max=x_max_absolute,
                                               y_min=y_min_absolute, y_max=y_max_absolute)
                    detection = Detection(label=label, score=score, bounding_box=bounding_box)
                    detections.append(detection)

        # save detections as text
        detection_lines = []
        for detection in detections:
            detection_line = detection_to_line_mapper(detection=detection)
            detection_lines.append(detection_line)
        save_file(path=detection_file_path, lines=detection_lines, mode='text')
        if log:
            print(detection_file_path, 'is saved.')

        # draw detections on image
        detection_image = np.copy(image_data)
        for detection in detections:
            if detection.score > threshold['draw']:
                detection_image = draw_bounding_box(image_data=detection_image, bounding_box=detection.bounding_box,
                                                    label=detection.label, score=detection.score,
                                                    color=color['red'], thickness=thickness)
        cv2.imwrite(detection_image_file_path, detection_image)
        if log:
            print(detection_image_file_path, 'is saved.')


if __name__ == '__main__':
    main()
