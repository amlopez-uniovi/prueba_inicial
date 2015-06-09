# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.signal as sgn
import matplotlib.pyplot as plt

def segment_signal2(S, ordenfiltro = 1, vecindad = 1, useacceleration = True, fcorte = 0.5):
    """
    Segmentación sobre un grupo de señales pasadas en una lista
    """
        
    tam = S[0].size
    
    h = np.zeros(tam)    
    b,a = sgn.butter(5, fcorte, 'lowpass', output = 'ba')
   
    
    for senhal in S:
        
        filtrada = sgn.filtfilt(b, a, senhal)
     
        diff1 = np.append(0, np.diff(filtrada))
         
        plt.plot(senhal)
        
        h = h +diff1**2
        if useacceleration:
           diff2 = np.append(0, np.diff(diff1))
           h = h + diff2**2    
        
    locmin = sgn.argrelextrema(h, np.less, order = vecindad)
    
    return (locmin[0], h)


