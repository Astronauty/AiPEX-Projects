import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dynamics(params, x, u):
    # cartpole ODE, parametrized by params. 

    # cartpole physical parameters 
    mc, mp, l = params.mc, params.mp, params.l
    g = 9.81
    
    q = x[:2]
    qd = x[2:]

    s = torch.sin(q[1])
    c = torch.cos(q[1])

    # print("q:" , q)
    # print(mc)
    # print(mp)
    # print(l)
    # print(s)
    # print(c)
    # print([mc+mp, mp*l*c])
    # print([mp*l*c, mp*l**2])
    
    # H = np.array([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]])
    # H = np.vstack([np.hstack([mc+mp, mp*l*c]), np.hstack([mp*l*c, mp*l**2])])
    H = torch.tensor([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]], device=device, requires_grad=True)

    # C = np.array([[0, -mp*qd[1]*l*s], [0, 0]])
    # C = np.vstack([np.hstack([0, -mp*qd[1]*l*s]), np.hstack([0, 0])])
    C = torch.tensor([[0, -mp*qd[1]*l*s], [0, 0]], device=device, requires_grad=True)

    # G = np.array([0, mp*g*l*s])
    # G = np.hstack([0, mp*g*l*s])
    G = torch.tensor([0, mp*g*l*s], device=device, requires_grad=True)

    # B = np.array([1, 0])
    B = torch.tensor([1, 0], device=device)

    # qdd = -np.linalg.inv(H) @ (C @ qd + G - B * u[1])
    
    # qdd = -np.linalg.solve(H, C @ qd + G - B * u[0])
    # qdd = -H\(C*qd + G - B*u[1])
    qdd = torch.linalg.solve(H, C @ qd + G - B * u[0])

    # xdot = np.hstack((qd, qdd)).T
    xdot = torch.cat((qd, qdd)).T
    return xdot 


def hermite_simpson(params, x1, x2, u, dt):
    x_m = 0.5*(x1 + x2) + (dt/8.0)*(dynamics(params, x1, u) - dynamics(params, x2, u))
    res = x1 + (dt/6) * (dynamics(params, x1, u) + 4*dynamics(params, x_m, u) + dynamics(params, x2, u)) - x2

    return res


def cartpole_dynamics_constraints(params, Z):
    idx, N, dt = params.idx, params.N, params.dt
    
    # Z = Z.detach().cpu().numpy()
    Z = Z.flatten()

    # TODO: create dynamics constraints using hermite simpson 

    # create c in a ForwardDiff friendly way (check HW0)
    # c = np.zeros(idx.nc, dtype=np.double)
    c = torch.zeros(idx.nc, dtype=torch.double, device=device)

    for i in range(0, N-1):
        xi = Z[idx.X[i]]
        ui = Z[idx.U[i]] 
        xip1 = Z[idx.X[i+1]]

        # print('xi: ', xi)   
        # print('ui: ', ui)
        
        # TODO: hermite simpson 
        c[idx.c[i]] = hermite_simpson(params, xi, xip1, ui, dt)

    # l2_norm = np.linalg.norm(c)
    # l2_norm = torch.norm(torch.tensor(c))
    l2_norm = torch.mean(c**2)
    # return c 
    return l2_norm


# def cartpole_equality_constraint(params, Z):
#     N, idx, xic, xg = params.N, params.idx, params.xic, params.xg 
    
    
#     # TODO: return all of the equality constraints 

    
#     return np.hstack([Z[idx.x[1]] - xic; Z[idx.x[end]] - xg,cartpole_dynamics_constraints(params, Z)])

