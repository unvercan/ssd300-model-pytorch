import types

from encodings.aliases import aliases


def convert_instance_to_text(instance):
    if not isinstance(instance, object):
        raise ValueError('"instance": parameter should be an object.')
    fields = []
    callable_instances = [types.FunctionType, types.MethodType]
    for key, value in instance.__dict__.items():
        is_callable = False
        for callable_instance in callable_instances:
            if isinstance(value, callable_instance):
                is_callable = True
        if not is_callable:
            fields.append('{key}={value}'.format(key=key, value=repr(value)))
    return '[' + ', '.join(fields) + ']'


def text_processor(text):
    if not isinstance(text, str):
        raise ValueError('"text": parameter should be a string value.')
    processed_text = text
    processed_text = processed_text.strip('\r\n')
    processed_text = processed_text.lower()
    return processed_text


def check_encoding_supported(encoding):
    if not isinstance(encoding, str):
        raise ValueError('"encoding": parameter should be a string value.')
    is_supported = False
    for key, value in enumerate(aliases.items()):
        if encoding in list(value):
            is_supported = True
            break
    return is_supported
