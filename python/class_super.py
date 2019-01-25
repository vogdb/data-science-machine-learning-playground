class SomeBaseClass(object):
    def __init__(self):
        print('SomeBaseClass.__init__(self) called')


class UnsuperChild(SomeBaseClass):
    def __init__(self):
        print('UnsuperChild.__init__(self) called')
        SomeBaseClass.__init__(self)


class SuperChild(SomeBaseClass):
    def __init__(self):
        print('SuperChild.__init__(self) called')
        # python2/3
        # super(SuperChild, self).__init__()
        # python3
        super().__init__()


class InjectMe(SomeBaseClass):
    def __init__(self):
        print('InjectMe.__init__(self) called')
        # python2/3
        # super(InjectMe, self).__init__()
        # python3
        super().__init__()


class UnsuperInjector(UnsuperChild, InjectMe):
    pass


class SuperInjector(SuperChild, InjectMe):
    pass


print('### simple example when no difference')
unsuper_child = UnsuperChild()
super_child = SuperChild()
print('\n### simple example with difference')
unsuper_injector = UnsuperInjector()
super_injector = SuperInjector()
