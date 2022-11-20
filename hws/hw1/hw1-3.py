import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import empirical_covariance
#%matplotlib inline

fig, ax = plt.subplots()

# n=100
np.random.seed(1234)

n = 100
epsilon = 1e-4
X = np.random.normal(size=(n,2))
result_mat_100 = np.zeros((n,2))

beta_0 = np.array([-2, 1]).reshape(2,1)
theta_0 = np.dot(X, beta_0)
p_0 = 1 / (1 + np.exp(-theta_0))

W_0 = np.diag(np.array([p_0[i][0] * (1-p_0[i][0]) for i in range(len(p_0))]))
I = np.dot(np.dot(X.T, W_0), X)

beta_i, beta_j = np.meshgrid(np.linspace(-4, 0, 1000), np.linspace(-0.5, 2.5, 1000))
f = (2 * np.pi * np.sqrt(np.linalg.norm(np.linalg.inv(I)))) ** (-1) * np.exp(-0.5 * ((beta_i+2) ** 2 * I[0,0] + 2 * (beta_i+2) * (beta_j-1) * I[0,1] + (beta_j-1) ** 2 * I[1,1]))

f_log = np.log10(f)
levels = np.linspace(f_log.min(), f_log.max(), 15)

ax.contour(beta_i, beta_j, f_log, levels=levels)

for iter in range(100):
    np.random.seed()
    Y = np.array([1 if np.random.geometric(p_0[i]) == 1 else 0 for i in range(len(p_0))]).reshape(n,1)

    beta = np.zeros(2).reshape(2,1)
    p = np.ones(n).reshape(n,1) / 2
    delta_beta= epsilon * 2

    W = np.diag(np.array([p[i][0] * (1-p[i][0]) for i in range(len(p))]))

    while delta_beta > epsilon:
        W = np.diag(np.array([p[i][0] * (1-p[i][0]) for i in range(len(p))]))
        beta_init = beta.copy()
        beta += np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), np.dot(X.T, (Y-p)))
        delta_beta = np.linalg.norm(beta - beta_init)
        p = 1 / (1 + np.exp(-np.dot(X, beta)))

    result_mat_100[iter] = beta.T


Cov_mat_100 = empirical_covariance(result_mat_100)
print(Cov_mat_100)
print(np.linalg.norm(Cov_mat_100))


# n=10000
np.random.seed(1234)

n = 10000
epsilon = 1e-4
X = np.random.normal(size=(n,2))
result_mat_10000 = np.zeros((n,2))

beta_0 = np.array([-2, 1]).reshape(2,1)
theta_0 = np.dot(X, beta_0)
p_0 = 1 / (1 + np.exp(-theta_0))

for iter in range(100):
    np.random.seed()
    Y = np.array([1 if np.random.geometric(p_0[i]) == 1 else 0 for i in range(len(p_0))]).reshape(n,1)

    beta = np.zeros(2).reshape(2,1)
    p = np.ones(n).reshape(n,1) / 2
    delta_beta= epsilon * 2

    W = np.diag(np.array([p[i][0] * (1-p[i][0]) for i in range(len(p))]))

    while delta_beta > epsilon:
        W = np.diag(np.array([p[i][0] * (1-p[i][0]) for i in range(len(p))]))
        beta_init = beta.copy()
        beta += np.dot(np.linalg.inv(np.dot(np.dot(X.T, W), X)), np.dot(X.T, (Y-p)))
        delta_beta = np.linalg.norm(beta - beta_init)
        p = 1 / (1 + np.exp(-np.dot(X, beta)))

    result_mat_10000[iter] = beta.T
    ax.scatter(beta[0][0], beta[1][0], c='tab:red', s=5)


Cov_mat_10000 = empirical_covariance(result_mat_10000)
print(Cov_mat_10000)
print(np.linalg.norm(Cov_mat_10000))


# plot settings
plt.rcParams.update({
    "text.usetex": True
})
fig.set_size_inches(8,6)
plt.savefig('3-4.jpg',dpi=1000, bbox_inches='tight')
plt.xlabel('$\\beta_i$')
plt.ylabel('$\\beta_j$')
plt.show()
