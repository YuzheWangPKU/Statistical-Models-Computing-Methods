import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ite
#%matplotlib inline

data = np.load('banana_shape_data.npy')
y = data.reshape(len(data),1)


def U(theta, y=y):
    """
    calculate the potential energy U(theta)
    """
    theta_1, theta_2 = theta[0], theta[1]
    U = 0.125 * np.sum(np.square(y - theta_1 - theta_2**2)) + 0.5 * (theta_1**2 + theta_2**2)

    return U


def K(r):
    """
    calculate the kinetic energy K(r)
    """
    K = 0.5 * np.dot(r.T, r)

    return K


def nabla_U(theta, y=y):
    """
    calculate the gradient of the potential energy U(theta)
    """
    theta_1, theta_2 = theta[0], theta[1]
    n_sum = np.sum(y - theta_1 - theta_2**2)
    nabla_U_1 = -0.25 * n_sum + theta_1
    nabla_U_2 = -0.5 * theta_2 * n_sum + theta_2
    nabla_U = np.array([nabla_U_1, nabla_U_2]).reshape(2,1)

    return nabla_U


def nabla_K(r):
    """
    calculate the gradient of the kinetic energy K(r)
    """
    return r


def H(theta, r, y=y):
    """
    calculate the Hamiltonian
    """
    return U(theta, y=y) + K(r)


def accept_p(beta_new, r_new, beta_init, r_init, y=y):
    """
    calculate the acceptance probability
    """
    a = min(1, np.exp(-H(beta_new, r_new, y=y) + H(beta_init, r_init, y=y)))

    return a


def leap_frog(beta, r, epsilon, y=y):
    """
    perform one step of leap-frog algorithm
    """
    r_m = r - 0.5 * epsilon * nabla_U(beta, y=y)
    beta_new = beta + epsilon * nabla_K(r_m)
    r_new = r_m - 0.5 * epsilon * nabla_U(beta_new, y=y)

    return beta_new, r_new


def HMC(L, sample_num, burn_in_num, epsilon, beta_0=np.zeros((2,1)), rand_steps=False, y=y):
    """
    implement Hamiltonian Monte Carlo algorithm\n
    if rand_steps is set to False (default), use a fixed L=L; otherwise use a random L ~ Uniform(1,L)
    """
    beta = beta_0
    beta_list = beta.copy()
    sample_collected = 0

    with tqdm(total = sample_num + burn_in_num) as pbar:
        while sample_collected < sample_num + burn_in_num:
            beta_init = beta.copy()
            r = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.eye(2)).reshape(2,1)
            r_init = r.copy()

            if rand_steps == False:
                L = L
            elif rand_steps == True:
                L = np.random.randint(low=1, high=L+1)

            for step in range(L):
                beta, r = leap_frog(beta, r, epsilon=epsilon, y=y)

            u = np.random.rand()
            if u < accept_p(beta_new=beta, r_new=r, beta_init=beta_init, r_init=r_init, y=y):
                sample_collected += 1
                pbar.update(1)
                if sample_collected > burn_in_num:
                    beta_list = np.concatenate((beta_list, beta.copy()), axis=1)
            else:
                beta = beta_init.copy()
                r = r_init.copy()

    return beta_list


# initialize and run the HMC sampler
theta_list = HMC(L=100, sample_num=500, burn_in_num=500, epsilon=0.05, rand_steps=False)


# plot scatter
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.scatter(theta_list[0][1:], theta_list[1][1:], s=10)

fig.set_size_inches(8,6)
plt.xlabel('$\\theta_i$', fontsize='x-large')
plt.ylabel('$\\theta_j$', fontsize='x-large')
plt.savefig('4-3-3.jpg',dpi=1000, bbox_inches='tight')
plt.show()
