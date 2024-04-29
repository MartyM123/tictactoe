def sigmoid(x):
  ''' vrátí hodnotu od 0 do 1 '''
  return 1 / (1 + np.exp(-x))

def plain(x):
    return x

def add_random_n_places(a : np.ndarray, n : int) -> np.ndarray:
    '''add random values to random n elements'''
    out = a.astype(float)

    # Generate unique flattened indices along the size of a
    idx = np.random.choice(a.size, n, replace=False)
    # Assign into those places ramdom numbers in [-1,1)
    out.flat[idx] += np.random.uniform(low=-1, high=1, size=n)
    return out

def get_max_exc(a : np.ndarray, exc=[]) -> int:
    '''return index of max element except indecies in exc:list'''
    m = np.zeros(a.size, dtype=bool)
    m[exc] = True
    b = np.ma.array(a, mask=m)
    return int(np.argmax(b))