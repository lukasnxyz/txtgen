from itertools import chain

def n_params(m):
    np = 0
    for p in list(m.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        np += nn
    return np