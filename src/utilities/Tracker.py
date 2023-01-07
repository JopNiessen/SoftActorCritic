"""

"""


class Tracker:
    def __init__(self, item_list):
        self.memory = {}
        for item in item_list:
            self.memory[item] = list()
    
    def add_item(self, key, item):
        self.memory[key].append(item)
    
    def add(self, item_list, key_list=list()):
        if len(item_list) != len(key_list):
            key_list = self.memory.keys()
        for key, item in zip(key_list, item_list):
            self.add_item(key, item)
    
    def __call__(self, key):
        return self.memory[key]
