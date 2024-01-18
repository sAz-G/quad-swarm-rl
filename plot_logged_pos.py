import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams['axes.grid'] = True

if __name__ == '__main__':

    pths = ['/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-42-42/Pose-20240109T20-50-17.csv',
            #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-42-42/Pose-20240109T20-51-01.csv'
            ]

    arr_pos = np.zeros((655, 3))

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
                rw = np.array(rw[1:4]).astype(float)

                arr_pos[k-1,:] = rw


                k = k + 1

        u = u+1

    #print(arr_thrst)


    fig, ax = plt.subplots(1,1)
    ax.plot(arr_pos[:,1],arr_pos[:,0])
    #ax.plot(arr_pos[:,1])
    #ax.plot(arr_pos[:,2])
    ax.set_xlabel("y")
    ax.set_ylabel("x")

    plt.show()
