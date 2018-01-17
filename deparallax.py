##########################################################################
#
# deparallax.py, version 0.4
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
# v0.3: correct the elongation calculation with proper great-circle distance;
#       force R to be an array
# v0.4: testing: project the coordinate to low latitude 
##########################################################################

import numpy as np
from skyfield.api import load, Topos
planets = load('de423.bsp')

def arc(RA1, RA2, DEC1, DEC2):
    dDEC = DEC1-DEC2
    dRA = RA1-RA2
    c = (np.sin((dDEC/2.))**2 + np.cos(DEC1)*np.cos(DEC2)*np.sin(dRA/2.)**2)**0.5
    ARC = 2*np.arcsin(c)
    return ARC
    
def topo_to_bary(tri_pos, ra_mean, dec_mean):
    ra = np.array(tri_pos[0])
    dec = np.array(tri_pos[1])
    mjd = np.array(tri_pos[2])
    R = np.array(tri_pos[3])
    mjd[np.isnan(mjd)] = 50000
    ts = load.timescale()
    t = ts.tai(jd=mjd+2400000.500428)
    earth = planets['earth']
    ctio = earth + Topos('30.169 S', '70.804 W', elevation_m=2200)
    x_earth, y_earth, z_earth = ctio.at(t).position.au #ICRS xyz of earth
    earth_distance = (x_earth**2 + y_earth**2 + z_earth**2)**0.5
    dec_bary = np.arcsin(-z_earth/earth_distance)
    ra_bary = np.arctan2(-y_earth, -x_earth)
    elong = arc(ra, ra_bary, dec, dec_bary)
    delta = earth_distance*np.cos(elong) + (R**2- (earth_distance*np.sin(elong))**2)**0.5
    x = np.cos(dec)*np.cos(ra)*delta #equatorial xyz of object
    y = np.cos(dec)*np.sin(ra)*delta
    z = np.sin(dec)*delta   
    xS = x + x_earth
    yS = y + y_earth
    zS = z + z_earth
    Rp = (xS**2+yS**2+zS**2)**0.5
    decS = np.arcsin(zS/Rp)
    raS = np.arctan2(yS,xS)%(2*np.pi)
    return raS, decS
    #ra_mean = ra.mean()
    #dec_mean = dec.mean()
    
    #midraS, middecS = raS.mean(), decS.mean()
    #midraS, middecS = ra_mean, dec_mean
    #lat_off = np.radians(5.0)
    #xaxis, yaxis, zaxis = (0, 0), (np.pi/2, 0), (0, np.pi/2)
    #nxaxis, nyaxis, nzaxis = ((midraS-np.pi/2)%(2*np.pi), 0), (midraS, middecS), ((midraS+np.pi)%(2*np.pi), (np.pi/2.0-middecS))
    #nx = np.cos(arc(nxaxis[0], xaxis[0], nxaxis[1], xaxis[1]))*xS + np.cos(arc(nxaxis[0], yaxis[0], nxaxis[1], yaxis[1]))*yS + np.cos(arc(nxaxis[0], zaxis[0], nxaxis[1], zaxis[1]))*zS
    #ny = np.cos(arc(nyaxis[0], xaxis[0], nyaxis[1], xaxis[1]))*xS + np.cos(arc(nyaxis[0], yaxis[0], nyaxis[1], yaxis[1]))*yS + np.cos(arc(nyaxis[0], zaxis[0], nyaxis[1], zaxis[1]))*zS
    #nz = np.cos(arc(nzaxis[0], xaxis[0], nzaxis[1], xaxis[1]))*xS + np.cos(arc(nzaxis[0], yaxis[0], nzaxis[1], yaxis[1]))*yS + np.cos(arc(nzaxis[0], zaxis[0], nzaxis[1], zaxis[1]))*zS
    #ndecS = np.arcsin(nz/Rp)
    #nraS = np.arctan2(ny,nx)%(2*np.pi)
    #return nraS, ndecS
    
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
