# -*- coding: utf-8 -*-

_global_dict={}
def _init():

    _global_dict = {}

def set_value(name, value):
    _global_dict[name] = value


def get_value(name, defValue=None):
    try:
        return _global_dict[name]
    except KeyError:
        return ""

