from multiprocessing import Pool

def f(x,**kwargs):
    print(f'kwargs {kwargs}')
    print(f"x {x}")
    print('ok')

if __name__ == '__main__':
    var = {'a':1,'b':2}
    with Pool(5) as p:
        p.map(f, [*{'a':1,'b':2}, *{'a':1,'b':2}])