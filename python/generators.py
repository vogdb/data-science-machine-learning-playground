def generate_list_of_odds():
    odds = [n for n in range(10) if n % 2 == 1]
    print('array', odds)
    odds = (n for n in range(10) if n % 2 == 1)
    print('generator', odds)
    print('unpacked generator', [n for n in odds])


def generate_list_of_pairs():
    odds = [n for n in range(10) if n % 2 == 1]
    evens = [n for n in range(10) if n % 2 == 0]
    pairs = ((o, e) for o, e in zip(odds, evens))
    print(next(pairs))


def generator_class_example():
    def firstn_fn(n):
        num, nums = 0, []
        while num < n:
            nums.append(num)
            num += 1
        return nums

    class Firstn_cl(object):
        def __init__(self, n):
            self.n = n
            self.num = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.num < self.n:
                current = self.num
                self.num += 1
                return current
            else:
                raise StopIteration()

    def firstn_fn_gen(n):
        num = 0
        while num < n:
            yield num
            num += 1

    print(firstn_fn(3), sum(firstn_fn(3)))
    print(Firstn_cl(3), sum(Firstn_cl(3)))
    print(firstn_fn_gen(3), sum(firstn_fn_gen(3)))


def iterable_emptying():
    class Firstn_cl(object):
        def __init__(self, n):
            self.n = n
            self.num = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.num < self.n:
                current = self.num
                self.num += 1
                return current
            else:
                raise StopIteration()

    def firstn_fn_gen(n):
        num = 0
        while num < n:
            yield num
            num += 1

    n_list = Firstn_cl(5)
    print('Class generator feature')
    print('first call is fine', list(n_list))
    print('second call is empty', list(n_list))
    n_list = firstn_fn_gen(5)
    print('\nFunction generator feature')
    print('first call is fine', list(n_list))
    print('second call is empty', list(n_list))


iterable_emptying()
