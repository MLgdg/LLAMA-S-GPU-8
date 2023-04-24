import json

class DictToClass(object):
    def __init__(self, _obj):
        if _obj:
            self.__dict__.update(_obj