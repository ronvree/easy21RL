

class ZeroDict(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, item):
        return self.get(item, 0)
