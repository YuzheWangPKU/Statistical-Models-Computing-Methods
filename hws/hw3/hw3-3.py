import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
#%matplotlib inline

observed_data = np.load('absolute_gaussian_data.npy')


def cal_l(y, mu, sigma2, gamma):
    """
    calculate the marginal log-likelihood function l(mu, sigma^2)
    """
    m = len(y)

    Q = -(((np.dot(y, y) + m*(mu**2) -2*mu*np.dot(2*gamma-1, y)) / (2*sigma2)) + m*np.log(2*np.pi*sigma2))
    H = np.dot(gamma, np.log(gamma)) + np.dot(1-gamma, np.log(1-gamma))
    l = Q - H

    return l


def EM_algorithm(y=observed_data, mu_0=1, sigma2_0=1, iter=20):
    """
    implement EM algorithm
    """
    l_list = []
    m = len(y)

    for i in range(iter):
        gamma = 1 / (1 + (np.exp(-np.square(y+mu_0)/(2*sigma2_0))) / (np.exp(-np.square(y-mu_0)/(2*sigma2_0))))
        mu = np.dot(2*gamma-1, y) / m
        sigma2 = (np.dot(y, y) + m*(mu**2) - 2*mu*np.dot(2*gamma-1, y)) / (2*m)

        l = cal_l(y=y, mu=mu, sigma2=sigma2, gamma=gamma)
        l_list.append(l)
        
        mu_0 = mu
        sigma2_0 = sigma2

    return l_list


def gradient_descent(y=observed_data, mu_0=1, sigma2_0=1, iter=20, eta=0.1):
    """
    implement standard gradient descent algorithm
    """
    l_list = []
    m = len(y)

    for i in range(iter):
        gamma = 1 / (1 + (np.exp(-np.square(y+mu_0)/(2*sigma2_0))) / (np.exp(-np.square(y-mu_0)/(2*sigma2_0))))
        mu_gradient = -(m*mu_0 - np.dot(2*gamma-1, y)) / (sigma2_0)
        sigma2_gradient = ((np.dot(y, y) + m*(mu_0**2) -2*mu_0*np.dot(2*gamma-1, y)) / (2*sigma2_0**2)) - (m / sigma2_0)

        mu = mu_0 + eta * mu_gradient
        sigma2 = sigma2_0 + eta * sigma2_gradient

        l = cal_l(y=y, mu=mu, sigma2=sigma2, gamma=gamma)
        l_list.append(l)

        mu_0 = mu
        sigma2_0 = sigma2

    return l_list


l_list_EM = EM_algorithm(y=observed_data, mu_0=1, sigma2_0=1, iter=20)
l_list_GD = gradient_descent(y=observed_data, mu_0=1, sigma2_0=1, iter=20, eta=0.008)
l_star = l_list_EM[-1]

# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})

ax.plot(np.arange(1,21), l_star-np.array(l_list_EM), label='$\\rm EM\\, algorithm$')
ax.plot(np.arange(1,21), l_star-np.array(l_list_GD), label='$\\rm standard\\, gradient\\, descent$')


fig.set_size_inches(8,6)
ax.xaxis.set_major_locator(MultipleLocator(2))
plt.xlabel('$\\rm iter$')
plt.ylabel('$l^{*}-l$')
plt.legend()
plt.savefig('3-2-2.jpg',dpi=1000, bbox_inches='tight')
plt.show()
