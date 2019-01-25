class Mapping:
    def __init__(self, iterable):
        self.items_list = []
        self.__update(iterable)

    def update(self, iterable):
        for item in iterable:
            self.items_list.append(item)

    __update = update

    def __print(self):
        print('Mapping')


class MappingSubclass(Mapping):

    def update(self, keys, values):
        # provides new signature for update()
        # but does not break __init__()
        for item in zip(keys, values):
            self.items_list.append(item)

    def __update(self):
        print('independent child\'s update')

    def __print(self):
        super()
        print('MappingSubclass')


print(vars(Mapping))
print(vars(MappingSubclass))
