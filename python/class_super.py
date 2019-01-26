def advantage_of_super_over_direct():
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


def mro_examples():
    class Root:
        def draw(self):
            # the delegation chain stops here
            assert not hasattr(super(), 'draw'), 'ancestor in _mro_ should inherit from {}'.format(Root)

    class Shape(Root):
        def __init__(self, shapename, **kwds):
            self.shapename = shapename
            super().__init__(**kwds)

        def draw(self):
            print('Drawing.  Setting shape to:', self.shapename)
            super().draw()

    class ColoredShape(Shape):
        def __init__(self, color, **kwds):
            self.color = color
            super().__init__(**kwds)

        def draw(self):
            print('Drawing.  Setting color to:', self.color)
            super().draw()

    def match_arguments_strategy():
        cs = ColoredShape(color='red', shapename='circle')
        print(cs.shapename, cs.color)

    def break_ancestor_chain():
        class Point():
            def draw(self):
                print('Breaking of ancestor\'s chain. Point dot insteadof Drawing.')

        class ColoredPoint1(ColoredShape, Point):
            pass

        class ColoredPoint2(Point, ColoredShape):
            pass

        print(ColoredPoint1.__mro__)
        try:
            cs = ColoredPoint1(color='blue', shapename='point')
            cs.draw()
        except AssertionError as e:
            print('Defense worked:\n\t', e)

        print(ColoredPoint2.__mro__)
        cs = ColoredPoint2(color='blue', shapename='point')
        cs.draw()

    def injector_adapter():
        class Moveable:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def draw(self):
                print('Drawing at position:', self.x, self.y)

        class MoveableAdapter(Root):
            def __init__(self, x, y, **kwds):
                self.moveable = Moveable(x, y)
                super().__init__(**kwds)

            def draw(self):
                self.moveable.draw()
                super().draw()

        class MoveableColoredShape(MoveableAdapter, ColoredShape):
            pass

        print(MoveableColoredShape.__mro__)
        triangle = MoveableColoredShape(color='red', shapename='triangle', x=1, y=2)
        triangle.draw()

    # match_arguments_strategy()
    injector_adapter()


# advantage_of_super_over_direct()
mro_examples()
