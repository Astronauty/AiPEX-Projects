import numpy as np
from collections import namedtuple

# class TrajectoryUtils():
#     def __init__(self, trajectory, nx, nu, N):
#         self.Z = Z
#         pass

#     create_idx(nx ,)

def create_idx(nx,nu,N):
    nz = (N-1) * nu + N * nx
    X = [i * (nx + nu) + np.arange(nx) for i in range(N)]
    U = [i * (nx + nu) + nx + np.arange(nu) for i in range(N-1)]

    nc = (N-1) * nx
    c = [i * nx + np.arange(nx) for i in range(N-1)]

    return namedtuple('idx', ['nx', 'nu', 'N', 'nz', 'nc', 'X', 'U', 'c'])(nx=nx, nu=nu, N=N, nz=nz, nc=nc, X=X, U=U, c=c)

# np.arange(8)
# nx=3
# nu=2
# N=2

# idx = create_idx(nx,nu,N)
# print(idx.c[0])

