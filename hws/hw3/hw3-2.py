import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

observed_data = np.load('absolute_gaussian_data.npy')
print(observed_data)




















n = len(logistic_data)
X = logistic_data[:,0:2]
y = logistic_data[:,2].reshape(n,1)


def U(beta, X=X, y=y):
    """
    calculate the potential energy U(beta)
    """
    U = -np.dot(np.dot(beta.T, X.T), (y - np.ones((n,1)))) + np.dot(np.ones((1,n)), np.log(1 + np.exp(-np.dot(X, beta)))) + 0.5 * np.dot(beta.T, beta)

    return U


def K(r):
    """
    calculate the kinetic energy K(r)
    """
    K = 0.5 * np.dot(r.T, r)

    return K


def nabla_U(beta, X=X, y=y):
    """
    calculate the gradient of the potential energy U(beta)
    """
    nabla_U = -np.dot(X.T, y - np.ones((n,1)) + np.exp(-np.dot(X, beta)) / (1 + np.exp(-np.dot(X, beta)))) + beta

    return nabla_U


def nabla_K(r):
    """
    calculate the gradient of the kinetic energy K(r)
    """
    return r


def H(beta, r, X=X, y=y):
    """
    calculate the Hamiltonian
    """
    return U(beta, X=X, y=y) + K(r)


def accept_p(beta_new, r_new, beta_init, r_init, X=X, y=y):
    """
    calculate the acceptance probability
    """
    a = min(1, np.exp(-H(beta_new, r_new, X=X, y=y) + H(beta_init, r_init, X=X, y=y)))

    return a


def leap_frog(beta, r, epsilon, X=X, y=y):
    """
    perform one step of leap-frog algorithm
    """
    r_m = r - 0.5 * epsilon * nabla_U(beta, X=X, y=y)
    beta_new = beta + epsilon * nabla_K(r_m)
    r_new = r_m - 0.5 * epsilon * nabla_U(beta_new, X=X, y=y)

    return beta_new, r_new


def HMC(L, sample_num, burn_in_num, epsilon, beta_0=np.zeros((2,1)), rand_steps=False, X=X, y=y):
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
                beta, r = leap_frog(beta, r, epsilon=epsilon, X=X, y=y)

            u = np.random.rand()
            if u < accept_p(beta_new=beta, r_new=r, beta_init=beta_init, r_init=r_init, X=X, y=y):
                sample_collected += 1
                pbar.update(1)
                if sample_collected > burn_in_num:
                    beta_list = np.concatenate((beta_list, beta.copy()), axis=1)
            else:
                beta = beta_init.copy()
                r = r_init.copy()

    return beta_list


# initialize and run the HMC sampler
beta_list = HMC(L=10, sample_num=500, burn_in_num=500, epsilon=0.01, rand_steps=False)


# plot scatter
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.scatter(beta_list[0][1:], beta_list[1][1:], s=10)

fig.set_size_inches(8,6)
plt.xlabel('$\\beta_i$')
plt.ylabel('$\\beta_j$')
plt.savefig('2-4-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()
