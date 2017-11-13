##########################################################################
#
# deparallax.py, version 0.1
#
# Remove the parallax motion by transferring the topographic equatorial
# coordinate to barycentric equatorial coordinate. 
#
# Author: 
# Edward Lin hsingwel@umich.edu
##########################################################################

import numpy as np
from skyfield.api import load

planets = load('de423.bsp')

def topo_to_bary(tri_pos):
    ra = tri_pos[0]
    dec = tri_pos[1]
    mjd = tri_pos[2]
    R = tri_pos[3]
    #print ra[4], dec[4], mjd[4], R[4]
    ts = load.timescale()
    t = ts.tai(jd=mjd+2400000.500428)
    earth = planets['earth']
    x_earth, y_earth, z_earth = earth.at(t).position.au
    earth_distance = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
    dec_bary = np.arcsin(-z_earth/earth_distance)
    ra_bary = np.arctan2(-y_earth, -x_earth)
    #print ra_bary, dec_bary
    elong = ((ra-ra_bary)**2+(dec-dec_bary)**2)**0.5
    delta = earth_distance*np.cos(elong) + (R**2- (earth_distance*np.sin(elong))**2)**0.5
    x = np.cos(dec)*np.cos(ra)*delta
    y = np.cos(dec)*np.sin(ra)*delta
    z = np.sin(dec)*delta
    xS = x + x_earth
    yS = y + y_earth
    zS = z + z_earth
    Rp = (xS**2+yS**2+zS**2)**0.5
#   print R, delta, earth_distance, x,y,z, x_earth, y_earth, z_earth, xS, yS, zS, Rp
    decS = np.arcsin(zS/Rp)
    raS = np.arctan2(yS,xS)
    return raS, decS
    
if __name__ == '__main__':
    print 'here'