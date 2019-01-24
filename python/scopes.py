def scope_test():
    def do_local():
        spam = 'local spam'

    def do_nonlocal():
        nonlocal spam
        spam = 'nonlocal spam'

    def do_nonlocal2():
        spam = 'local spam'

        def do_nonlocal():
            nonlocal spam
            spam = 'nonlocal2 spam'

        do_nonlocal()
        print('nonlocal2 changes the closes inner nonlocal: ', spam)

    def do_global():
        global spam
        spam = 'global spam'

    spam = 'test spam'
    do_local()
    print('After local assignment:', spam)
    do_nonlocal()
    print('After nonlocal assignment:', spam)
    do_nonlocal2()
    print('After nonlocal2 assignment:', spam)
    do_global()
    print('After global assignment:', spam)


try:
    spam
except NameError:
    print('no spam var in global scope')

scope_test()
print('In global scope:', spam)
