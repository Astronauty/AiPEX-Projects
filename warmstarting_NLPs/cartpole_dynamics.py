import numpy as np

def dynamics(params, x, u):
    # cartpole ODE, parametrized by params. 

    # cartpole physical parameters 
    mc, mp, l = params.mc, params.mp, params.l
    g = 9.81
    
    q = x[:2]
    qd = x[2:]

    s = np.sin(q[1])
    c = np.cos(q[1])

    print(x.shape)
    # print(mc)
    # print(mp)
    # print(l)
    print(s)
    print(c)
    # print([mc+mp, mp*l*c])
    # print([mp*l*c, mp*l**2])
    H = np.array([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]])
    C = np.array([[0, -mp*qd[1]*l*s], [0, 0]])
    G = np.array([0, mp*g*l*s])
    B = np.array([1, 0])

    # qdd = -np.linalg.inv(H) @ (C @ qd + G - B * u[1])
    
    qdd = -np.linalg.solve(H, C @ qd + G - B * u[0])
    # qdd = -H\(C*qd + G - B*u[1])
    xdot = np.hstack((qd, qdd)).T
    return xdot 


def hermite_simpson(params, x1, x2, u, dt):
    x_m = 0.5*(x1 + x2) + (dt/8.0)*(dynamics(params, x1, u) - dynamics(params, x2, u))
    res = x1 + (dt/6) * (dynamics(params, x1, u) + 4*dynamics(params, x_m, u) + dynamics(params, x2, u)) - x2
    return res


def cartpole_dynamics_constraints(params, Z):
    idx, N, dt = params.idx, params.N, params.dt
    
    Z = Z.detach().cpu().numpy()
    # TODO: create dynamics constraints using hermite simpson 

    # create c in a ForwardDiff friendly way (check HW0)
    c = np.zeros(idx.nc, dtype=type(Z))
    
    for i in range(0, N-1):
        xi = Z[idx.X[i]]
        ui = Z[idx.U[i]] 
        xip1 = Z[idx.X[i+1]]
        
        # TODO: hermite simpson 
        c[idx.c[i]] = hermite_simpson(params, xi, xip1, ui, dt)

    return c 


# def cartpole_equality_constraint(params, Z):
#     N, idx, xic, xg = params.N, params.idx, params.xic, params.xg 
    
    
#     # TODO: return all of the equality constraints 

    
#     return np.hstack([Z[idx.x[1]] - xic; Z[idx.x[end]] - xg,cartpole_dynamics_constraints(params, Z)])

