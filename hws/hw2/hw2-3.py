import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
from tqdm import tqdm
#%matplotlib inline

probit_data = np.load('probit_data.npy')
n = len(probit_data)
x = probit_data[:,0].reshape(n,1)
y = probit_data[:,1].reshape(n,1)

# set the priors
beta_0, sigma2_0, tau2_0, nu_0 = 0., 1., 100., 3.


def beta_new(sigma2, z, x=x, beta_0=beta_0, tau2_0=tau2_0):
    """
    update beta
    """
    x2_sum = np.sum(x ** 2)
    x_z_sum = np.sum(x * z)
    beta_new = np.random.normal(loc = (beta_0 * sigma2 + tau2_0 * x_z_sum) / (sigma2 + tau2_0 * x2_sum), scale = np.sqrt((tau2_0 * sigma2) / (sigma2 + tau2_0 * x2_sum)))
   
    return beta_new


def sigma2_new(beta, z, x=x, n=n, nu_0=nu_0, sigma2_0=sigma2_0):
    """
    update sigma^2
    """
    square_sum = np.sum((z - x * beta) ** 2)
    sigma2_new = invgamma.rvs(a = (n + nu_0) / 2, scale = (nu_0 * sigma2_0 + square_sum) / 2)

    return sigma2_new


def z_i_new(i, sigma2, beta, x=x, y=y):
    """
    update z_i
    """
    x_i = x[i][0]
    y_i = y[i][0]
    mu_i = (x_i * beta) / (np.sqrt(sigma2))
    u_i = np.random.rand()

    if y_i == 1:
        z_i_new = x_i * beta + np.sqrt(sigma2) * norm.ppf(norm.cdf(-mu_i) + u_i * norm.cdf(mu_i))
    elif y_i == 0:
        z_i_new = x_i * beta + np.sqrt(sigma2) * norm.ppf(u_i * norm.cdf(-mu_i))

    return z_i_new


def true_posterior(beta, sigma2, n=n, beta_0=beta_0, sigma2_0=sigma2_0, tau2_0=tau2_0, nu_0=nu_0, x=x, y=y):
    """
    calculate the true posterior (unnormalized) of (beta, sigma^2)
    """
    beta_prior = norm.pdf(beta, loc=beta_0, scale=np.sqrt(tau2_0))
    sigma2_prior = invgamma.pdf(sigma2, a=0.5*nu_0, scale=0.5*nu_0*sigma2_0)
    prod = 1

    for i in range(n):
        x_i = x[i][0]
        y_i = y[i][0]
        prod *= (norm.cdf(x_i * beta / np.sqrt(sigma2)) ** y_i) * (1 - norm.cdf(x_i * beta / np.sqrt(sigma2))) ** (1 - y_i)

    posterior = beta_prior * sigma2_prior * prod

    return posterior


def Gibbs_sampler(num_iter, beta_init, sigma2_init, z_init, MH=False, beta_0=beta_0, sigma2_0=sigma2_0, tau2_0=tau2_0, nu_0=nu_0, x=x, y=y):
    """
    implement Gibbs sampler
    MH: when set True, insert a Metropolis-Hasting step after each Gibbs cycle that scales the current state of the Markov chain (default=False)
    """
    beta = beta_init
    sigma2 = sigma2_init
    z = z_init

    beta_list, sigma2_list = [beta], [sigma2]

    for iter in tqdm(range(num_iter)):
        # update beta
        beta = beta_new(sigma2=sigma2, z=z, x=x, beta_0=beta_0, tau2_0=tau2_0)
        beta_list.append(beta)

        # update sigma^2
        sigma2 = sigma2_new(beta=beta, z=z, x=x, n=n, nu_0=nu_0, sigma2_0=sigma2_0)
        sigma2_list.append(sigma2)

        # update z_i individually
        for i in range(n):
            z[i] = z_i_new(i=i, sigma2=sigma2, beta=beta, x=x, y=y)

        # Metropolis-Hastings step that scales the current state of the Markov chain
        if MH == True:
            s = np.random.exponential()
            a = min(1, (true_posterior(beta=s*beta, sigma2=s*sigma2) / true_posterior(beta=beta, sigma2=sigma2)) * np.exp(s - (1/s)))
            #print(a)
            u = np.random.rand()
            if u < a:
                beta = s * beta
                sigma2 = s * sigma2

    return beta_list, sigma2_list


# initialize the Gibbs sampler
beta_init, sigma2_init = 25, 25
z_init = np.zeros((n,1))
num_iter = 2000

# run the Gibbs sampler
beta_list, sigma2_list = Gibbs_sampler(num_iter=num_iter, beta_init=beta_init, sigma2_init=sigma2_init, z_init=z_init, MH=False)


# generate the trace plots of beta and sigma^2
fig, (ax1, ax2) = plt.subplots(2,1)
plt.rcParams.update({
    "text.usetex": True
})

ax1.plot(np.linspace(0, num_iter, num_iter+1), np.array(beta_list))
ax1.set_xlabel('$\\rm iter$')
ax1.set_ylabel('$\\beta$')

ax2.plot(np.linspace(0, num_iter, num_iter+1), np.array(sigma2_list))
ax2.set_xlabel('$\\rm iter$')
ax2.set_ylabel('$\\sigma^2$')

fig.set_size_inches(8,6)
plt.savefig('2-3-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()


# plot samples and contours
beta_grid, sigma2_grid = np.meshgrid(np.linspace(np.min(beta_list[1:]), np.max(beta_list[1:]), 1000), np.linspace(np.min(sigma2_list[1:]), np.max(sigma2_list[1:]), 1000))
posterior = true_posterior(beta_grid, sigma2_grid)
log_posterior = np.log(posterior + 1e-25)

fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.contour(beta_grid, sigma2_grid, log_posterior, levels=10)
ax.scatter(np.array(beta_list[1:]), np.array(sigma2_list[1:]), s=1)
fig.set_size_inches(8,6)
plt.xlabel('$\\beta$')
plt.ylabel('$\\sigma^2$')
plt.savefig('2-3-2.jpg',dpi=1000, bbox_inches='tight')
plt.show()

