import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.grid'] = True

if __name__ == '__main__':
    # pths = ['/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/nnthrust-trhst-20240109T17-31-58.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-pos-20240109T17-31-56.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-vel-20240109T17-31-54.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R1-20240109T17-31-55.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R2-20240109T17-31-55.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-R3-20240109T17-31-54.csv',
    #         '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T17-30-41/slfOb-w-20240109T17-32-01.csv'
    #         ]
    pths = ['/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/nnthrust-trhst-20240109T20-27-13.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-pos-20240109T20-27-07.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-R1-20240109T20-27-06.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-R2-20240109T20-27-06.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-R3-20240109T20-27-05.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-vel-20240109T20-27-05.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-26-21/slfOb-w-20240109T20-27-04.csv'
            ]

    # arr_thrst = np.zeros((869, 4))
    # arr_pos   = np.zeros((892, 3))
    # arr_vel   = np.zeros((909, 3))
    # arr_r1    = np.zeros((896, 3))
    # arr_r2    = np.zeros((901, 3))
    # arr_r3    = np.zeros((905, 3))
    # arr_w = np.zeros((905, 3))

    arr_thrst = np.zeros((1689, 4))
    arr_pos = np.zeros((1748, 3))
    arr_vel = np.zeros((1758, 3))
    arr_r1 = np.zeros((1760, 3))
    arr_r2 = np.zeros((1765, 3))
    arr_r3 = np.zeros((1770, 3))
    arr_w = np.zeros((1774, 3))

    u = 0
    for p in range(pths.__len__()):

        with open(pths[p]) as fl:
            k = 0
            for row in fl:
                if k == 0:
                    k = k+1
                    continue

                rw = row
                rw = rw.split(',')
                rw = np.array(rw[1:]).astype(float)

                if u == 0:
                    arr_thrst[k-1,:] = rw
                elif u == 1:
                    arr_pos[k-1,:] = rw
                elif u == 2:
                    arr_vel[k-1,:] = rw
                elif u == 3:
                    arr_r1[k-1,:] = rw
                elif u == 4:
                    arr_r2[k-1,:] = rw
                elif u == 5:
                    arr_r3[k-1,:] = rw
                elif u == 6:
                    arr_w[k-1,:] = rw

                k = k + 1

        u = u+1

    #print(arr_thrst)


    fig, ax = plt.subplots(7,1)
    ax[0].plot(arr_thrst[:,0])
    ax[0].plot(arr_thrst[:,1])
    ax[0].plot(arr_thrst[:,2])
    ax[0].plot(arr_thrst[:,3])
    #ax.legend(['m_0', 'm_1', 'm_2', 'm_3'])
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Motor Thrust")

    ax[1].plot(arr_pos[:, 0])
    ax[1].plot(arr_pos[:, 1])
    ax[1].plot(arr_pos[:, 2])
    # ax.legend(['m_0', 'm_1', 'm_2', 'm_3'])
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Motor Thrust")

    ax[2].plot(arr_vel[:, 0])
    ax[2].plot(arr_vel[:, 1])
    ax[2].plot(arr_vel[:, 2])
    # ax.legend(['m_0', 'm_1', 'm_2', 'm_3'])
    ax[2].set_xlabel("Samples")
    ax[2].set_ylabel("Motor Thrust")

    ax[3].plot(arr_r1[:, 0])
    ax[3].plot(arr_r1[:, 1])
    ax[3].plot(arr_r1[:, 2])
    ax[3].set_xlabel("Samples")
    ax[3].set_ylabel("Motor Thrust")

    ax[4].plot(arr_r2[:, 0])
    ax[4].plot(arr_r2[:, 1])
    ax[4].plot(arr_r2[:, 2])
    ax[4].set_xlabel("Samples")
    ax[4].set_ylabel("Motor Thrust")

    ax[5].plot(arr_r3[:, 0])
    ax[5].plot(arr_r3[:, 1])
    ax[5].plot(arr_r3[:, 2])
    ax[5].set_xlabel("Samples")
    ax[5].set_ylabel("Motor Thrust")

    ax[6].plot(arr_w[:, 0])
    ax[6].plot(arr_w[:, 1])
    ax[6].plot(arr_w[:, 2])
    ax[6].set_xlabel("Samples")
    ax[6].set_ylabel("Motor Thrust")
    plt.show()
