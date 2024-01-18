import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #arr = np.load('/home/saz/GitHub/quad-swarm-rl/actions_0.npy')
    #arr = np.load('/home/saz/GitHub/quadswarmsharif/poses/poses_3.npy')
    t = np.linspace(0, np.pi * 2, 300)
    a1 = 0.4 * np.cos(t) #+ np.random.randn(300) * 0.03
    a2 = 0.4 * np.sin(t) #+ np.random.randn(300) * 0.03
    circ = np.asarray([a1, a2])
    circ = circ.reshape(2,300)
    # m1 = arr[:1000,0]
    # m2 = arr[:1000,1]
    # m3 = arr[:1000,2]
    # m4 = arr[:1000,3]

    # mx = 0
    ax = plt.figure().add_subplot(projection='3d')


    # plt.plot(m1)
    # plt.plot(m2)
    # plt.plot(m3)
    #plt.plot(circ[0], circ[1])
    ax.plot(a1, a2, linewidth=1)
    #plt.plot(a1)
    plt.show()


