import numpy as np
import matplotlib.pyplot as plt

def f(x:float, y:float):
    """
    Beale's function
    """
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x - x * y ** 3) ** 2


def nabla_f(x:float, y:float):
    """
    gradient of f, returns a 2D array
    """
    fx = 2 * (-1 + y) * (1.5 - x + x * y) + 2 * (-1 + y ** 2) * (2.25 - x + x * y ** 2) + 2 * (-1 - y ** 3) * (2.625 - x - x * y ** 3)
    fy = 2 * x * (1.5 - x + x * y) + 4 * x * y * (2.25 - x + x * y ** 2) - 6 * x * y ** 2 * (2.625 - x - x * y ** 3)

    return np.array([fx, fy])


def noise_nabla_f(x:float, y:float, sigma=0.01):
    """
    stochastic gradient of f, returns a 2D array
    """
    fx = 2 * (-1 + y) * (1.5 - x + x * y) + 2 * (-1 + y ** 2) * (2.25 - x + x * y ** 2) + 2 * (-1 - y ** 3) * (2.625 - x - x * y ** 3)
    fy = 2 * x * (1.5 - x + x * y) + 4 * x * y * (2.25 - x + x * y ** 2) - 6 * x * y ** 2 * (2.625 - x - x * y ** 3)

    return np.random.normal(loc=np.array([fx, fy]), scale=sigma)


# gradient decent
def GD_vanilla(x0:float, y0:float, fmin=0.0135145, lr=0.01, iter=150):
    """
    implementation of vanilla gradient decent
    """
    z = np.array([x0, y0])
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        z += -lr * nabla_f(z[0], z[1])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)
    
    return result_list      


# gradient decent with momentum
def GD_momentum(x0:float, y0:float, fmin=0.0135145, mu=0.85, lr=0.01, iter=150):
    """
    implementation of gradient decent with momentum
    """
    z = np.array([x0, y0])
    m = nabla_f(x0, y0)
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        z += -lr * m
        m = mu * m + (1-mu) * nabla_f(z[0], z[1])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)
        
    return result_list


# Nesterov's accelerated gradient decent
def GD_nesterov(x0:float, y0:float, fmin=0.0135145, lr=0.01, iter=150):
    """
    implementation of Nesterov's accelerated gradient decent
    """
    z = np.array([x0, y0])
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        if i == 0:
            z += -lr * nabla_f(z[0], z[1])
            coord_list.append((i+1, z.copy()))
            result_list.append(f(z[0], z[1])-fmin)
        else:
            y = coord_list[i][1] + (coord_list[i][1] - coord_list[i-1][1]) * (i - 1) / (i + 2)
            z = y - lr * nabla_f(y[0], y[1])
            coord_list.append((i+1, z.copy()))
            result_list.append(f(z[0], z[1])-fmin)

    return result_list


# stochastic gradient decent
def SGD_vanilla(x0:float, y0:float, fmin=0.0135145, lr=0.01, iter=150):
    """
    implementation of vanilla stochastic gradient decent
    """
    z = np.array([x0, y0])
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        z += -lr * noise_nabla_f(z[0], z[1])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)
    
    return result_list


# AdaGrad
def SGD_AdaGrad(x0:float, y0:float, fmin=0.0135145, epsilon=1e-8, lr=1, iter=150):
    """
    implementation of AdaGrad
    """
    z = np.array([x0, y0])
    coord_list = [(0, z.copy())]
    gx2_sum, gy2_sum = 0, 0 # g refers to gradient
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        (gx, gy) = noise_nabla_f(z[0], z[1])
        gx2_sum += gx ** 2
        gy2_sum += gy ** 2
        z += -lr * np.array([gx / np.sqrt(gx2_sum + epsilon), gy / np.sqrt(gy2_sum + epsilon)])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)
        
    return result_list

# RMSprop
def SGD_RMSprop(x0:float, y0:float, fmin=0.0135145, epsilon=1e-8, beta=0.98, lr=0.06, iter=150):
    """
    implementation of RMSprop
    """
    z = np.array([x0, y0])
    gx2_sum = gy2_sum = 0
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        (gx, gy) = noise_nabla_f(z[0], z[1])
        gx2_sum = beta * gx2_sum + (1-beta) * gx ** 2
        gy2_sum = beta * gy2_sum + (1-beta) * gy ** 2
        z += -lr * np.array([gx / np.sqrt(gx2_sum + epsilon), gy / np.sqrt(gy2_sum + epsilon)])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)

    return result_list


# Adam
def SGD_Adam(x0:float, y0:float, fmin=0.0135145, epsilon=1e-8, beta_1=0.75, beta_2=0.97, lr=0.07, iter=150):
    """
    implementation of Adam
    """
    z = np.array([x0, y0])
    mx = my = vx = vy = 0
    coord_list = [(0, z.copy())]
    result_list = [f(x0, y0)-fmin]

    for i in range(iter):
        (gx, gy) = noise_nabla_f(z[0], z[1])
        mx = beta_1 * mx + (1-beta_1) * gx 
        my = beta_1 * my + (1-beta_1) * gy
        vx = beta_2 * vx + (1-beta_2) * gx ** 2
        vy = beta_2 * vy + (1-beta_2) * gy ** 2

        mx_hat = mx / (1 - beta_1 ** (i+1))
        my_hat = my / (1 - beta_1 ** (i+1))
        vx_hat = vx / (1 - beta_2 ** (i+1))
        vy_hat = vy / (1 - beta_2 ** (i+1))

        z += -lr * np.array([mx_hat / np.sqrt(vx_hat + epsilon), my_hat / np.sqrt(vy_hat + epsilon)])
        coord_list.append((i+1, z.copy()))
        result_list.append(f(z[0], z[1])-fmin)

    return result_list



#print(GD_vanilla(0.5, 2)[-1])
#print(GD_momentum(0.5, 2)[-1])
#print(GD_nesterov(0.5, 2)[-1])
print(SGD_vanilla(0.5, 2)[-1])
print(SGD_AdaGrad(0.5, 2)[-1])
print(SGD_RMSprop(0.5, 2)[-1])
print(SGD_Adam(0.5, 2)[-1])

k = np.linspace(0, 150, 151)
fig, ax = plt.subplots()
"""
ax.plot(k, np.array(GD_vanilla(0.5, 2)))
ax.plot(k, np.array(GD_momentum(0.5, 2)))
ax.plot(k, np.array(GD_nesterov(0.5, 2)))
"""
ax.plot(k, np.array(SGD_vanilla(0.5, 2)))
ax.plot(k, np.array(SGD_AdaGrad(0.5, 2)))
ax.plot(k, np.array(SGD_RMSprop(0.5, 2)))
ax.plot(k, np.array(SGD_Adam(0.5, 2)))

plt.yscale('log')
plt.rcParams.update({
    "text.usetex": True
})
plt.xlabel('$k$')
plt.ylabel('$f-f^*$')
#plt.legend(['$\\rm vanilla \\, GD$', '\\rm momentum', '\\rm Nesterov'])
plt.legend(['$\\rm vanilla \\, SGD$', '\\rm AdaGrad', '\\rm RMSprop', '\\rm Adam'])
plt.show()
