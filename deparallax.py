##########################################################################
#
# deparallax.py, version 0.3
#
# Remove the TNO parallax motion by transferring the topographic equatorial
# coordinate to barycentric equatorial coordinate. 
#
# Author: 
# Edward Lin hsingwel@umich.edu
#
# v0.2: added parallax_distance function
# v0.2.1: fixed negative RA issue
# v0.2.2: fixed 'nan' mjd issue
# v0.3: correct the elongation calculation; force R to be an array 
##########################################################################

import numpy as np
from skyfield.api import load
planets = load('de423.bsp')

def arc(RA1, RA2, DEC1, DEC2):
    a = np.cos(np.pi/2.- DEC1)*np.cos(np.pi/2.- DEC2)
    b = np.sin(np.pi/2.- DEC1)*np.sin(np.pi/2.- DEC2) * np.cos(RA1-RA2)
    return np.arccos(a+b)

def topo_to_bary(tri_pos):
    ra = np.array(tri_pos[0])
    dec = np.array(tri_pos[1])
    mjd = np.array(tri_pos[2])
    R = np.array(tri_pos[3])
    mjd[np.isnan(mjd)] = 50000
    #print ra[4], dec[4], mjd[4], R[4]
    ts = load.timescale()
    t = ts.tai(jd=mjd+2400000.500428)
    earth = planets['earth']
    x_earth, y_earth, z_earth = earth.at(t).position.au
    earth_distance = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
    dec_bary = np.arcsin(-z_earth/earth_distance)
    ra_bary = np.arctan2(-y_earth, -x_earth)
    #print ra_bary, dec_bary
    #elong = ((ra-ra_bary)**2+(dec-dec_bary)**2)**0.5
    elong = arc(ra, ra_bary, dec, dec_bary)
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
    raS[raS < 0] += 2*np.pi
    raS[raS > 2*np.pi] -= 2*np.pi
    return raS, decS
    
def parallax_distance(ra1, ra2, dec1, dec2, mjd1, mjd2):
    ts = load.timescale()
    mjd1[np.isnan(mjd1)] = 50000
    mjd2[np.isnan(mjd2)] = 50000
    t1 = ts.tai(jd=mjd1+2400000.500428)
    t2 = ts.tai(jd=mjd2+2400000.500428)
    earth = planets['earth']      
    x_earth1, y_earth1, z_earth1 = earth.at(t1).position.au
    x_earth2, y_earth2, z_earth2 = earth.at(t2).position.au
    vx = x_earth2-x_earth1
    vy = y_earth2-y_earth1
    vz = z_earth2-z_earth1  
    velo = (vx**2+vy**2+vz**2)**0.5
    dec_v = np.arcsin(vz/velo)
    ra_v = np.arctan2(vy, vx)
    x = np.cos(dec1)*np.cos(ra1)
    y = np.cos(dec1)*np.sin(ra1)
    z = np.sin(dec1)
    theta = ((ra2-ra1)**2 + (dec2-dec1)**2)**0.5
    vec1 = zip(x, y, z)
    vec2 = zip(vx, vy, vz)
    ab = abs(np.array((map(np.inner, vec1, vec2))))
    cosphi = ab/velo
    phi = ((ra1-ra_v)**2+(dec1-dec_v)**2)**0.5
    parallax_dis = (1-cosphi**2)**0.5*velo/theta
    return parallax_dis
    
if __name__ == '__main__':
    print 'here'
