class NonNegative:
    def __get__(self, instance, owner):
        print('NonNegative.__get__ instance {}, owner {}'.format(instance, owner))
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        print('NonNegative.__set__ instance {}, value {}'.format(instance, value))
        if value < 0:
            raise ValueError('Cannot be negative.')
        instance.__dict__[self.name] = value

    def __set_name__(self, owner, name):
        print('NonNegative.__set_name__ owner {}, name {}'.format(owner, name))
        self.name = name


class Order:
    price = NonNegative()
    quantity = NonNegative()

    def __init__(self, name, price, quantity):
        self._name = name
        self.price = price
        self.quantity = quantity

    def total(self):
        return self.price * self.quantity


apple_order = Order('apple', 1, 10)
apple_order.total()
# 5
apple_order.price = 5
# 10
apple_order.price = -10
# ValueError: Cannot be negative
apple_order.quantity = -10
# ValueError: Cannot be negative
