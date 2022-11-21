import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

x = np.array([1.41, 1.84, 1.64, 0.85, 1.32, 1.97, 1.70, 1.02, 1.84, 0.92], dtype=float)
r = np.array([0.94, 0.70, 0.16, 0.38, 0.40, 0.57, 0.24, 0.27, 0.60, 0.81], dtype=float)
y = np.array([13, 17, 6, 3, 7, 13, 8, 7, 5, 8], dtype=int)

theta_0 = 1
epsilon = 1e-5
err = 1 + epsilon
theta_list = [theta_0]

while err >= epsilon:
    theta = (theta_0 / np.sum(x)) * (np.dot(x / (x * theta_0 + r), y))
    err = np.abs(theta - theta_0)
    theta_list.append(theta)
    theta_0 = theta

# plot
fig, ax = plt.subplots()
plt.rcParams.update({
    "text.usetex": True
})
ax.plot(np.arange(0, len(theta_list)), np.array(theta_list))

fig.set_size_inches(8,6)
plt.xlabel('$\\rm iter$')
plt.ylabel('$\\theta$')
plt.savefig('3-1-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()

print(f'the MLE of theta is {theta}')

I_obs = np.dot(np.square(x / (theta * x + r)), y)
print(f'the observed Fisher information is {I_obs}')

I_com = np.dot(x / theta, y / (x * theta + r))
print(f'the complete information is {I_com}')

frac = 1 - I_obs / I_com
print(f'the fraction of missing information is {frac}')
