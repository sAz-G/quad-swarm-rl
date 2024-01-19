import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = np.linspace(0, 1, 1000)

    x0 = 0.5
    print(np.max(np.abs(np.exp(x0) + np.exp(x0)*(x-x0) + np.exp(x0)*(x-x0)**2/2 + np.exp(x0)*(x-x0)**3/6 - np.exp(x))))
    print(np.max(np.abs(np.exp(x0) + np.exp(x0)*(x-x0) + np.exp(x0)*(x-x0)**2/2 - np.exp(x))))
    print(np.max(np.abs(np.exp(x0) + np.exp(x0)*(x-x0) - np.exp(x))))
    plt.plot(x, np.exp(x))
    plt.plot(x, np.exp(x0) + np.exp(x0)*(x-x0) + np.exp(x0)*(x-x0)**2/2)
    plt.plot(x, np.exp(x0) + np.exp(x0)*(x-x0) + np.exp(x0)*(x-x0)**2/2 + np.exp(x0)*(x-x0)**3/6 )
    #plt.plot(x, np.exp(0.2) + np.exp(0.2)*(x-0.2))
    #plt.plot(x, np.exp(0.8) + np.exp(0.8)*(x-0.8))
    plt.show()