import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams['axes.grid'] = True

if __name__ == '__main__':

    pths = [#'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-17-39/Pose-20240121T14-19-12.csv',
            '/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240109T20-42-42/Pose-20240109T20-51-01.csv'
            #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-28-59/Pose-20240121T14-31-09.csv'
            #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-28-59/Pose-20240121T14-31-03.csv'
            #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-28-59/Pose-20240121T14-30-47.csv'
        #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-28-59/Pose-20240121T14-30-17.csv'
        #'/home/saz/GitHub/quad-swarm-rl/logged_cfclient/20240121T14-28-59/Pose-20240121T14-29-23.csv'
        #'/home/saz/GitHub/quadswarmsharif/logged_cfclient/20240121T15-08-29/Pose-20240121T15-13-03.csv'
        #'/home/saz/GitHub/quadswarmsharif/logged_cfclient/20240121T15-08-29/Pose-20240121T15-12-17.csv'
        #'/home/saz/GitHub/quadswarmsharif/logged_cfclient/20240121T15-08-29/Pose-20240121T15-11-59.csv'
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

    x = np.linspace(0,1,1000)


    fig, ax = plt.subplots(1,1)
    ax.plot(arr_pos[:,1],arr_pos[:,0])
    #ax.plot(.5*np.cos(2*np.pi*x),.5*np.sin(2*np.pi*x), '--')
    #ax.plot(arr_pos[:,1])
    #ax.plot(arr_pos[:,2])
    ax.set_xlabel("y")
    ax.set_ylabel("x")

    plt.show()
