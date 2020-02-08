import os
import types

from common.helpers import check_encoding_supported


def read_file(path, mode='text', encoding='utf_8', processor=None):
    if not isinstance(path, str):
        raise ValueError('"path": parameter should be a string value.')
    elif not os.path.exists(path):
        raise ValueError('"path": file does not exist.')
    if mode == 'text':
        read_mode = 'r'
    elif mode == 'binary':
        read_mode = 'rb'
    else:
        raise ValueError('"mode": parameter can be "text" or "binary".')
    if not check_encoding_supported(encoding):
        raise ValueError('"encoding": unsupported encoding.')
    if not (isinstance(processor, types.FunctionType) or processor is None):
        raise ValueError('"encoding": parameter can be a function or None value.')
    lines = []
    with open(file=path, mode=read_mode, encoding=encoding) as file:
        for _, line in enumerate(file):
            if processor is not None:
                line_processed = processor(line)
                lines.append(line_processed)
            else:
                lines.append(line)
    return lines


def save_file(path, lines, mode='text', encoding='utf_8'):
    if not isinstance(path, str):
        raise ValueError('"path": parameter should be a string value.')
    if not isinstance(lines, list):
        raise ValueError('"lines": parameter should be a list.')
    if mode == 'text':
        write_mode = 'w'
    elif mode == 'binary':
        write_mode = 'wb'
    else:
        raise ValueError('"mode": parameter can be "text" or "binary".')
    if not check_encoding_supported(encoding):
        raise ValueError('"encoding": unsupported encoding.')
    with open(file=path, mode=write_mode, encoding=encoding) as file:
        for _, line in enumerate(lines):
            if not isinstance(line, str):
                raise ValueError('"lines": parameter should be a list of strings.')
            line += '\n'
            file.write(line)


def get_file_paths_in_directory(directory):
    if not isinstance(directory, str):
        raise ValueError('"directory": parameter should be a string value.')
    file_paths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths


def filter_file_paths_by_extensions(file_paths, extensions):
    if not isinstance(file_paths, list):
        raise ValueError('"file_paths": parameter should be a list.')
    if not isinstance(extensions, list):
        raise ValueError('"extensions": parameter should be a list.')
    if len(extensions) == 0:
        return file_paths
    filtered_file_paths = []
    for file_path in file_paths:
        extension = file_path.split('\\')[-1].split('.')[-1]
        if extension in extensions:
            if not isinstance(extension, str):
                raise ValueError('"extensions": parameter should be a list of strings.')
            filtered_file_paths.append(file_path)
    return filtered_file_paths
