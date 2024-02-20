import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid'] = False

plt.rcParams.update({
     "text.usetex": True,
     "font.family": "Serif"
})
plt.rcParams.update({'font.size': 17})
if __name__ == '__main__':

    pths = [
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-18-45.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T15-08-29/Pose-20240121T15-11-42.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-19-12.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240109T20-42-42/Pose-20240109T20-50-17.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-18-45.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T15-08-29/Pose-20240121T15-11-42.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-30-17.csv'

        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240109T20-42-42/Pose-20240109T20-50-17.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-18-19.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-18-45.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-17-39/Pose-20240121T14-19-12.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-29-23.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-29-44.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-30-17.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-30-47.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T14-28-59/Pose-20240121T14-31-03.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T15-08-29/Pose-20240121T15-10-47.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240121T15-08-29/Pose-20240121T15-11-42.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240109T20-42-42/Pose-20240109T20-51-01.csv'
        #'/home/saz/Desktop/qsrl_test/logged_cfclient/20240109T20-42-42/Pose-20240109T20-51-01.csv'
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
    #ax.plot(arr_pos[:325,1],arr_pos[:325,0])
    ax.plot(arr_pos[150:-350,1],arr_pos[150:-350,0])
    #ax.plot(arr_pos[110:-90,1],arr_pos[80:-120,0])
    #ax.plot([-0.85, -0.85], [1, 0.3], '--', color='0.45')
    #ax.plot([-0.85, 1], [1, 1], '--', color='0.45')
    #ax.plot([0.3, -0.6], [-0.8, -0.8], '--', color='0.45')
    #ax.plot([-0.6, -0.6], [-0.8, 0.3], '--', color='0.45')
    #ax.plot([0.3, -0.6], [0.3, 0.3], '--', color='0.45')


    #ax.plot([arr_pos[0,1], -0.6], [arr_pos[0,0], -0.5], '--', color='0.45')
    #ax.plot([arr_pos[0,1], -0.6], [arr_pos[0,0], -0.5], '--', color='0.45')
    #ax.plot([arr_pos[0,1], -0.6], [arr_pos[0,0], -0.5], '--', color='0.45')
    #ax.plot([-0.6, 0.5], [-0.5, -1], '--',  color='0.45')
   # ax.plot(0.5*np.cos(2*np.pi*x),0.5*np.sin(2*np.pi*x), '--',  color='0.45')
    #ax.plot(arr_pos[:,1])
    #ax.plot(arr_pos[:,2])
    ax.set_xlabel("$y$ (m)")
    ax.set_ylabel("$x$ (m)")

    plt.show()
