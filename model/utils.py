from itertools import chain

def n_params(m):
    np = 0
    for p in list(m.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
    return np

def flatten_list(n_list: list):
    return list(chain(*n_list))
