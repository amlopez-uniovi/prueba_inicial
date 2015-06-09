import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import segmentation as sgm
import imp


imp.reload(sgm)

config_mat = sio.loadmat('./datos/config.mat', struct_as_record=False, squeeze_me=True)

silop_config_struct = config_mat['SILOP_CONFIG']

datos = np.loadtxt('./datos/datos.log')
datos_alg = np.loadtxt('./datos/datos_alg.log')
datos = np.concatenate((datos, datos_alg), axis = 1)

flex_munheca = datos[:, silop_config_struct.SENHALES.m.flex-1]
abd_munheca = datos[:, silop_config_struct.SENHALES.m.abd-1]

flex_codo = datos[:, silop_config_struct.SENHALES.c.flex-1]
pron_codo = datos[:, silop_config_struct.SENHALES.c.pron-1]

plt.figure(1)

sp1 = plt.subplot(311)

plt.title('Muñeca')

plt.plot(flex_munheca, 'r', label = "flex")
plt.plot(abd_munheca,'g', label = "abd")

plt.legend(loc='upper right')
plt.grid()

plt.subplot(312, sharex=sp1)

plt.title("Codo")

plt.plot(flex_codo, 'r', label = "flex")
plt.plot(pron_codo, 'b', label = "pron")

flex_hombro = datos[:, silop_config_struct.SENHALES.h.flex-1]
rot_hombro = datos[:, silop_config_struct.SENHALES.h.rot-1]
abd_hombro = datos[:, silop_config_struct.SENHALES.h.abd-1]


plt.legend(loc='upper right')
plt.grid()

plt.subplot(313, sharex=sp1)

plt.title("Hombro")

plt.plot(flex_hombro, 'r', label = "flex")

plt.plot(rot_hombro, 'm', label = "rot")

plt.plot(abd_hombro, 'g', label = "abd")


plt.legend(loc='upper right')

plt.show()
plt.grid()

res = sgm.segment_signal2([flex_munheca, abd_munheca, flex_codo, pron_codo, flex_hombro, rot_hombro, abd_hombro], 
                          vecindad = 10, useacceleration = False, ordenfiltro = 20, fcorte = 0.05)


eventos1 = res[0]
derivada = res[1]
#
#eventos2 = np.array(())
#
#k = 0
#
#for k in np.arange(eventos1.size-1):
#    subsenhal = derivada[eventos1[k]:eventos1[k+1]]
#    maximo = np.amax(subsenhal)
#    
#    if((maximo-derivada[eventos1[k]] > 2) and (derivada[eventos1[k]] < 1)):
#        eventos2 = np.append(eventos2, eventos1[k]);

eventos2 = np.array([t for t in eventos1 if derivada[t] < 1])

eventos3 = np.array([eventos2[t] for t in range(eventos2.size-1) if np.amax(derivada[eventos2[t]:eventos2[t+1]])>10])



fig2 = plt.figure(2)

#sens = np.array((flex_munheca, abd_munheca, flex_codo, pron_codo, flex_hombro, rot_hombro, abd_hombro))
#plt.plot(sens.transpose())

ax1 = fig2.add_subplot(111)

ax1.plot(flex_munheca, 'r', label = "flex_munheca")
ax1.plot(abd_munheca,'g', label = "abd_munheca")
ax1.plot(flex_codo, 'b', label = "flex_codo")
ax1.plot(pron_codo, 'y', label = "pron_codo")
ax1.plot(flex_hombro, 'k', label = "flex_hombro")
ax1.plot(rot_hombro, 'm', label = "rot_hombro")
ax1.plot(abd_hombro, 'c', label = "abd_hombro")
ax1.legend(loc='upper right')
ax1.set_ylabel("Señales")
ax1.grid()


ax2 = fig2.add_subplot(111, sharex=ax1, frameon=False)
ax2.plot(derivada)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel("veloc-acel ")

ax2.bar(eventos1, np.ones(eventos1.size)*20)

ax2.plot(eventos2, np.ones(eventos2.size)*(-1), '*')

ax2.bar(eventos3, np.ones(eventos3.size)*(-20))


ax2.grid()

"""
Voy a codificar las señales de alguna manera para despues clasificarlas

En principio esto se trata de un simple juego y voy a ir comparando unas con otras para ver quienes se parecen más entre si
"""


golpes = {}

for k in np.arange(eventos3.size-1):
    
    sens = np.array((flex_munheca[eventos3[k]:eventos3[k+1]], 
                     abd_munheca[eventos3[k]:eventos3[k+1]], 
                     flex_codo[eventos3[k]:eventos3[k+1]], 
                     pron_codo[eventos3[k]:eventos3[k+1]], 
                     flex_hombro[eventos3[k]:eventos3[k+1]], 
                     rot_hombro[eventos3[k]:eventos3[k+1]], 
                     abd_hombro[eventos3[k]:eventos3[k+1]]   ))
                     
    golpes.update({k:(eventos3[k], eventos3[k+1], sens)})



import multi_dtw as mdtw

masparecido = {}

for k in np.arange(len(golpes)):
    distancia_minima = float('inf')
    print('K = ', k)
    for r in np.arange(len(golpes)):
        print('    R = ', r)
        if(k!=r):
            distancia, cost, path = mdtw.multi_dtw(golpes[k][2], golpes[r][2], normalize = False)
            if(distancia < distancia_minima):
                distancia_minima = distancia
                masparecido[k] = r
        


