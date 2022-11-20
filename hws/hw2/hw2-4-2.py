import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
from tqdm import tqdm
import ite
#%matplotlib inline

logistic_data = np.load('mcs_hw2_p3_data.npy')
n = len(logistic_data)
X = logistic_data[:,0:2]
y = logistic_data[:,2].reshape(n,1)

beta_ground_truth = np.load('ground_truth.npy')


def draw_batch(n_batch, n=n, X=X, y=y):
    """
    draw a batch (batch size = n_batch) of data items randomly from the raw data
    """
    batch_id = np.arange(n)
    np.random.shuffle(batch_id)
    X_batch = X[batch_id[:n_batch],:]
    y_batch = y[batch_id[:n_batch],:]

    return X_batch, y_batch


def KL_div(data, ground_truth=beta_ground_truth):
    """
    calculate KL divergence between the given data distribution and the ground-truth distribution
    """
    co = ite.cost.BDKL_KnnK()
    d = co.estimation(data.T, ground_truth.T)
    
    return d


def g(beta, n_batch, n=n, X=X, y=y):
    """
    calculate g(beta) for SGLD\n
    n_batch: batch size\n
    """
    X_batch, y_batch = draw_batch(n_batch=n_batch, n=n, X=X, y=y) 

    g = -beta + (n / n_batch) * np.dot(X_batch.T, y_batch - np.ones((n_batch,1)) + np.exp(-np.dot(X_batch, beta)) / (1 + np.exp(-np.dot(X_batch, beta))))

    return g


def SGLD(iter_num, KL_start, n_batch, epsilon_0, lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth):
    """
    implement Stochastic Gradient Langevin Dynamics algorithm\n
    start to calculate KL divergence at KL_start-th iteration
    """
    beta = beta_0
    beta_list = beta.copy()
    KL_div_list = []

    for iter in tqdm(range(iter_num)):
        epsilon = epsilon_0 * np.exp(-lambda_0 * iter)
        eta = np.random.multivariate_normal(mean=np.array([0,0]), cov=epsilon*np.eye(2)).reshape(2,1)
        beta_new = beta + 0.5 * epsilon * g(beta=beta, n_batch=n_batch, n=n, X=X, y=y) + eta

        beta_list = np.concatenate((beta_list, beta_new.copy()), axis=1)
        if iter >= KL_start - 1:
            KL_div_list.append(KL_div(data=beta_list, ground_truth=ground_truth))
        
        beta = beta_new

    return KL_div_list


def SGHMC(iter_num, KL_start, n_batch, C, epsilon_0, lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth):
    """
    implement Stochastic Gradient Hamiltonian Monte Carlo algorithm\n
    start to calculate KL divergence at KL_start-th iteration
    """
    beta = beta_0
    r = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.eye(2)).reshape(2,1)
    beta_list = beta.copy()
    KL_div_list = []

    for iter in tqdm(range(iter_num)):
        epsilon = epsilon_0 * np.exp(-lambda_0 * iter)
        eta = np.random.multivariate_normal(mean=np.array([0,0]), cov=2*C*epsilon*np.eye(2)).reshape(2,1)

        beta_new = beta + epsilon * r
        r_new = r + epsilon * g(beta=beta, n_batch=n_batch, n=n, X=X, y=y) - epsilon * C * r + eta

        beta_list = np.concatenate((beta_list, beta_new.copy()), axis=1)
        if iter >= KL_start - 1:
            KL_div_list.append(KL_div(data=beta_list, ground_truth=ground_truth))
        
        beta = beta_new
        r = r_new

    return KL_div_list


def SGNHT(iter_num, KL_start, n_batch, A, epsilon_0, lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth):
    """
    implement Stochastic Gradient Nose-Hoover Thermostat algorithm\n
    start to calculate KL divergence at KL_start-th iteration
    """
    beta = beta_0
    r = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.eye(2)).reshape(2,1)
    xi = A

    beta_list = beta.copy()
    KL_div_list = []

    for iter in tqdm(range(iter_num)):
        epsilon = epsilon_0 * np.exp(-lambda_0 * iter)
        eta = np.sqrt(2 * A) * np.random.multivariate_normal(mean=np.array([0,0]), cov=epsilon*np.eye(2)).reshape(2,1)

        r_new = r + epsilon * g(beta=beta, n_batch=n_batch, n=n, X=X, y=y) - epsilon * xi * r + eta
        beta_new = beta + epsilon * r_new
        xi_new = xi + epsilon * (0.5 * np.dot(r_new.T, r_new) - 1)

        beta_list = np.concatenate((beta_list, beta_new.copy()), axis=1)
        if iter >= KL_start -1:
            KL_div_list.append(KL_div(data=beta_list, ground_truth=ground_truth))
        
        beta = beta_new
        r = r_new
        xi = xi_new

    return KL_div_list


# set the parameters of the SGMCMC samplers
iter_num = 1000
KL_start = 5
n_batch = 100


# run the SGLD sampler
epsilon_0 = 0.0075
lambda_0 = 0.01

KL_div_list_SGLD = SGLD(iter_num=iter_num, KL_start=KL_start, n_batch=n_batch, epsilon_0=epsilon_0, lambda_0=lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth)


# run the SGHMC sampler
epsilon_0 = 0.0075
c_0 = 80
lambda_0 = 1e-5

KL_div_list_SGHMC = SGHMC(iter_num=iter_num, KL_start=KL_start, n_batch=n_batch, C=c_0, epsilon_0=epsilon_0, lambda_0=lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth)


# run the SGNHT sampler
epsilon_0 = 0.0075
a_0 = 110
lambda_0 = 1e-5

KL_div_list_SGNHT = SGNHT(iter_num=iter_num, KL_start=KL_start, n_batch=n_batch, A=a_0, epsilon_0=epsilon_0, lambda_0=lambda_0, beta_0=np.zeros((2,1)), n=n, X=X, y=y, ground_truth=beta_ground_truth)


# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.plot(np.arange(KL_start, iter_num + 1), np.array(KL_div_list_SGLD), label='\\rm SGLD')
ax.plot(np.arange(KL_start, iter_num + 1), np.array(KL_div_list_SGHMC), label='\\rm SGHMC')
ax.plot(np.arange(KL_start, iter_num + 1), np.array(KL_div_list_SGNHT), label='\\rm SGNHT')

fig.set_size_inches(8,6)
plt.xlabel('$\\rm iter$')
plt.ylabel('$\\rm KL \, Divergence$')
plt.legend()
plt.savefig('2-4-4.jpg',dpi=1000, bbox_inches='tight')
plt.show()

