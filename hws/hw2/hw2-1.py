import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

def f(x:float):
    """
    PDF of the standard Laplace distribution
    """
    return 0.5 * np.exp(-np.abs(x))

def g(x:float):
    """
    PDF of the the standard normal distribution
    """
    return np.exp(- x**2 / 2) / np.sqrt(2 * np.pi)

def F_inv(x:float):
    """
    inverse function of the CDF of the standard Laplace distribution
    """
    assert (x >= 0 and x <= 1), "invalid input for F_inv"
    if x < 0.5:
        return np.log(2 * x)
    else:
        return -np.log(2 - 2 * x)

c = np.sqrt(2 * np.e / np.pi)
x_list = []

while len(x_list) <= 10000:
    (u, v)=np.random.rand(2)
    x = F_inv(u)
    if v <= (g(x) / (c * f(x))):
        x_list.append(x)


fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.hist(np.array(x_list), density=True, bins=40, edgecolor='white', label='$\\rm Rejection\, Sampling$')
ax.plot(np.linspace(-4.5, 4.5, 1000), g(np.linspace(-4.5, 4.5, 1000)), label='$\\mathcal{N}(0,1)$')

# plot settings
fig.set_size_inches(8,6)
plt.xlabel('$x$')
plt.ylabel('$p(x)$')
plt.legend()
plt.savefig('2-1-2.jpg',dpi=1000, bbox_inches='tight')
plt.show()
