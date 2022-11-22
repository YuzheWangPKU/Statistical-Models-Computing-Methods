import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

observed_data = np.load('absolute_gaussian_data.npy')


def EM_algorithm(y=observed_data, mu_0=0, sigma2_0=1, epsilon=1e-5):
    """
    implement EM algorithm
    """
    mu_list, sigma2_list = [mu_0], [sigma2_0]
    m = len(y)
    err = 1 + epsilon

    while err >= epsilon:
        gamma = 1 / (1 + (np.exp(-np.square(y+mu_0)/(2*sigma2_0))) / (np.exp(-np.square(y-mu_0)/(2*sigma2_0))))
        mu = np.dot(2*gamma-1, y) / m
        sigma2 = (np.dot(y, y) + m*(mu**2) - 2*mu*np.dot(2*gamma-1, y)) / (2*m)

        err = max(np.abs(mu - mu_0), np.abs(sigma2-sigma2_0))
        mu_list.append(mu)
        sigma2_list.append(sigma2)

        mu_0 = mu
        sigma2_0 = sigma2

    return mu_list, sigma2_list


# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})

for mu_0 in np.linspace(4, -4, 21):
    for sigma2_0 in np.linspace(0.1, 5, 11):
        mu_list, sigma2_list = EM_algorithm(y=observed_data, mu_0=mu_0, sigma2_0=sigma2_0, epsilon=1e-5)
        ax.plot(np.array(mu_list), np.array(sigma2_list), color='tab:blue')
        ax.scatter(mu_list[-1], sigma2_list[-1], color='tab:orange', zorder=2)

fig.set_size_inches(8,6)
plt.xlabel('$\\mu$')
plt.ylabel('$\\sigma^2$')
plt.savefig('3-2-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()


