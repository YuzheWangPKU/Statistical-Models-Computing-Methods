import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.special import digamma
#%matplotlib inline

with open('btc_hw4_lda_data.p', 'rb') as handle:
    data_loaded = pickle.load(handle)

data = data_loaded['data']
beta_matrix = data_loaded['beta_matrix']


def VI_LDA(i, D_mat=data, B_mat=beta_matrix, alpha=0.1, epsilon=1e-3):
    """
    implement variational inference algorithm on human ancestry discovery i.e. LDA model for a single individual
    i: id of individual (start from 1)
    D_mat: data matrix
    B_mat: beta matrix
    """
    (N, K) = B_mat.shape
    d_vec = D_mat[i-1]
    d_vec_valid = d_vec[d_vec != 0]
    B_mat_valid = B_mat[d_vec != 0]
    N_i = np.sum(d_vec_valid, dtype=int)
    N_valid = len(d_vec_valid)

    phi_mat_0 = np.ones(N_valid*K, dtype=float).reshape(N_valid, K) / K
    gamma_vec_0 = np.ones(K, dtype=float).reshape(K,1) * (alpha + N_i / K)
    err = 1 + epsilon
    num_iter = 0

    while err >= epsilon:
        num_iter += 1
        phi_mat = phi_mat_0.copy()
        for j in range(N_valid):
            phi_vec_j = B_mat_valid[j] * (np.exp(digamma(gamma_vec_0)).T)
            phi_mat[j] = phi_vec_j / np.sum(phi_vec_j)
        gamma_vec = np.ones(4, dtype=float).reshape(K,1) * alpha + np.dot(d_vec_valid, phi_mat).reshape(K,1)

        err = max(np.max(np.abs(phi_mat-phi_mat_0)), np.max(np.abs(gamma_vec-gamma_vec_0)))

        phi_mat_0 = phi_mat.copy()
        gamma_vec_0 = gamma_vec.copy()

    return phi_mat, gamma_vec, num_iter

# (2)
phi_1_mat, _, _ = VI_LDA(i=1, D_mat=data, B_mat=beta_matrix, alpha=0.1, epsilon=1e-3)
 
# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})

im = ax.imshow(phi_1_mat)
cbar = ax.figure.colorbar(im, shrink=0.3, pad=0.1)
cbar.ax.set_ylabel('$\\phi$', rotation=360, va='bottom')
ax.set_xticks(np.arange(len(phi_1_mat[0])), labels=np.arange(1, len(phi_1_mat[0])+1).tolist())
ax.set_yticks(np.arange(len(phi_1_mat[:,0]), step=5), labels=np.arange(1, len(phi_1_mat[:,0])+1, step=5).tolist())


fig.set_size_inches(4,16)
plt.xlabel('$\\rm ancestor$')
plt.ylabel('$\\rm genotype$')
plt.savefig('3-3-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# (3)
M, N, K = len(data[:,0]), len(data[0]), len(beta_matrix[0])
theta_mat = np.zeros(M*K).reshape(M, K)
num_iter_list = []

for i in range(M):
    _, gamma_vec, num_iter = VI_LDA(i+1, D_mat=data, B_mat=beta_matrix, alpha=0.1, epsilon=1e-3)
    theta_mat[i] = gamma_vec.reshape(1, K)
    num_iter_list.append(num_iter)

print(f'the maximum number of iterations needed to get to convergence for all {M} individuals is {max(num_iter_list)}')

# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})

im = ax.imshow(theta_mat)
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.3, pad=0.1)
cbar.ax.set_ylabel('$\\gamma$', rotation=360, va='bottom')
ax.set_xticks(np.arange(len(theta_mat[0])), labels=np.arange(1, len(theta_mat[0])+1).tolist())
ax.set_yticks(np.arange(len(theta_mat[:,0]), step=3), labels=np.arange(1, len(theta_mat[:,0])+1, step=3).tolist())


fig.set_size_inches(4,16)
plt.xlabel('$\\rm ancestor$')
plt.ylabel('$\\rm individual$')
plt.savefig('3-3-2.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# (4)
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.hist(np.array(num_iter_list), density=False, bins=40, edgecolor='white')

# plot settings
fig.set_size_inches(8,6)
plt.xlabel('$\\rm iterations\,for\,convergence$')
plt.ylabel('$\\rm number\, of\, individuals$')
plt.savefig('3-3-3.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# (5)
M, N, K = len(data[:,0]), len(data[0]), len(beta_matrix[0])

# plot
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
plt.rcParams.update({
    "text.usetex": True
})

for s, alpha in enumerate([0.01, 0.1, 1, 10]):
    theta_mat = np.zeros(M*K).reshape(M, K)
    num_iter_list = []

    for i in range(M):
        _, gamma_vec, num_iter = VI_LDA(i+1, D_mat=data, B_mat=beta_matrix, alpha=alpha, epsilon=1e-3)
        theta_mat[i] = gamma_vec.reshape(1, K)
        num_iter_list.append(num_iter)

    print(f'the mean number of iterations needed to get to convergence for all {M} individuals is {np.mean(num_iter_list)} (alpha={alpha})')

    ax = eval('ax'+f'{s+1}')
    im = ax.imshow(theta_mat)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.3, pad=0.1)
    cbar.ax.set_ylabel('$\\gamma$', rotation=360, va='bottom')
    ax.set_xticks(np.arange(len(theta_mat[0])), labels=np.arange(1, len(theta_mat[0])+1).tolist())
    ax.set_yticks(np.arange(len(theta_mat[:,0]), step=3), labels=np.arange(1, len(theta_mat[:,0])+1, step=3).tolist())

    ax.set_xlabel('$\\rm ancestor$')
    ax.set_ylabel('$\\rm individual$')
    ax.set_title(f'$\\alpha={alpha}$')


fig.set_size_inches(16,16)
plt.savefig('3-3-4.jpg',dpi=1000, bbox_inches='tight')
plt.show()

