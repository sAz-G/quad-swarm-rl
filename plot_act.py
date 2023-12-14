import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    arr = np.load('/home/saz/GitHub/quad-swarm-rl/actions_0.npy')

    m1 = arr[:1000,0]
    m2 = arr[:1000,1]
    m3 = arr[:1000,2]
    m4 = arr[:1000,3]

    mx = 0



    plt.plot(m1)
    plt.plot(m2)
    plt.plot(m3)
    plt.plot(m4)

    plt.show()


