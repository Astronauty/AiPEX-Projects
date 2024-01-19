from scipy import *
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import time

# Initialization
tstart = 0
tstop = 20
increment = 0.01
# Initial condition
x_init = [0,0]
t = np.arange(tstart,tstop+1,increment)
# Function that returns dx/dt
def parallel_msd(x, t, c, k, m, F):
    F = 0
    dx1dt = x[1]
    dx2dt = (F - c*x[1] - k*(x[0]-1))/m
    dxdt = [dx1dt, dx2dt]
    return dxdt

def parallel_ms(x, t, c, k, m, F):
    F = 0
    dx1dt = x[1]
    dx2dt = (F - k*(x[0]-1))/m
    dxdt = [dx1dt, dx2dt]
    return dxdt


k_mean = 10
m_mean = 10
c_mean = 1.2*np.sqrt(m_mean*k_mean)

# Ground truth
x1_groundtruth = odeint(parallel_msd, x_init, t, args=(c_mean,k_mean,m_mean,5))[:,0]
x2_groundtruth = odeint(parallel_msd, x_init, t, args=(c_mean,k_mean,m_mean,5))[:,1]
# plt.plot(t,x1_groundtruth, linewidth=4, label=f'Ground Truth. c={c_mean:.2f}, k={k_mean:.2f}, m={m_mean:.2f}.') 
plt.plot(t,x2_groundtruth, linewidth=4, label=f'Ground Truth. c={c_mean:.2f}, k={k_mean:.2f}, m={m_mean:.2f}.') 
# Stochastic rollout
k_normal = np.random.normal(k_mean, 2.5, size=5)
m_normal = np.random.normal(m_mean, 2.5, size=5)
c_normal = np.random.normal(c_mean, 5, size=5)

start = time.time()
for i in range(5):
    c = c_normal[i]
    k = k_normal[i]
    m = m_normal[i] 
    # print(c,k,m)
    # Solve ODE
    x = odeint(parallel_msd, x_init, t, args=(c,k,m,5))
    x1 = x[:,0]
    # print(x1[5])
    x2 = x[:,1]
    # Plot the Results
    mse = mean_squared_error(x2, x2_groundtruth)
    # plt.plot(t,x1, label=f'Rollout {i}. k={k:.2f}, m={m:.2f}, MSE = {mse:.4f}')
    # plt.plot(t,x2, label=f'Rollout {i}. k={k:.2f}, m={m:.2f}, MSE = {mse:.4f}')
    # plt.plot(t,x1, label=f'Rollout {i}. k={k:.2f}, m={m:.2f}, c={c:0.2f}, MSE = {mse:.4f}')
    plt.plot(t,x2, label=f'Rollout {i}. k={k:.2f}, m={m:.2f}, c={c:0.2f}, MSE = {mse:.4f}')
    # plt.plot(t,x2)
end = time.time()
print(end-start)

plt.title('Stochastic Rollout: Parallel Mass-Spring Structure')
plt.xlabel('t')
plt.ylabel('x_dot(t)')
plt.legend(loc='lower right')
plt.grid()
plt.show()