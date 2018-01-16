##########################################################################
#
# two_pairs.py, version 0.1
#
# connect two different pairs to form a track
#
# Author: 
# Edward Lin: hsingwel@umich.edu
#
##########################################################################

from __future__ import division
import gc
import pandas as pd
import numpy as np
import ephem
from Orbit import Orbit
import matplotlib.pyplot as plt
import pickle, json
from deparallax import topo_to_bary, parallax_distance
import subprocess, glob
import time, sys, os


def load_linkmap(linkmap):
    return pickle.load(open(linkmap))
    
class pairs:
    def __init__(self, linkmap_file_list, det_file_list):
        self.linkmap_list = map(load_linkmap, linkmap_file_list)
        self.detections_list = map(pd.read_csv, det_file_list)
        self.pairs_list = map(self.gen_pair_list, self.linkmap_list)
        self.df_list = map(self.getDatePos, self.pairs_list, self.detections_list)
        self.df = pd.concat(self.df_list, ignore_index=True)

    def gen_pair_list(self, linkmap):
        pair_list = []
        for i in linkmap.keys():
            if linkmap[i] != []:
                for j in linkmap[i]:
                    pair_list.append([i, j])
                    
        return pair_list
                           
    def getDatePos(self, pair_list, detections):
        obj1 = pd.DataFrame(data = np.array(pair_list).T[0], columns = ['id'])
        obj2 = pd.DataFrame(data = np.array(pair_list).T[1], columns = ['id'])
        obj1 = obj1.join(detections.set_index('objid'), on='id')
        obj2 = obj2.join(detections.set_index('objid'), on='id')
        
        ra1 = obj1['ra'].values*np.pi/180.
        ra2 = obj2['ra'].values*np.pi/180.

        dec1 = obj1['dec'].values*np.pi/180.
        dec2 = obj2['dec'].values*np.pi/180.

        fakeid1 = obj1['fakeid'].values
        fakeid2 = obj2['fakeid'].values
        
        mjd1 = obj1['mjd'].values
        mjd2 = obj2['mjd'].values
        
        id1 = obj1['id'].values
        id2 = obj2['id'].values

        expnum1 = obj1['expnum'].values
        expnum2 = obj2['expnum'].values
        
        exptime1 = obj1['exptime'].values
        exptime2 = obj2['exptime'].values
        
        band1 = obj1['band'].values
        band2 = obj2['band'].values
        
        ccd1 = obj1['ccd'].values
        ccd2 = obj2['ccd'].values
        
        mag1 = obj1['mag'].values
        mag2 = obj2['mag'].values
        
        ml_score1 = obj1['ml_score'].values
        ml_score2 = obj2['ml_score'].values
        
        date1 = obj1['date'].values
        date2 = obj2['date'].values
        
        d_para = parallax_distance(ra1, ra2, dec1, dec2, mjd1, mjd2)
        
        pairs = np.array([ra1, ra2, dec1, dec2, mjd1, mjd2, date1, date2, fakeid1, fakeid2, id1, id2, expnum1, expnum2, \
                          exptime1, exptime2, band1, band2, ccd1, ccd2, mag1, mag2, ml_score1, ml_score2, \
                          d_para])
        columns = ['ra1', 'ra2', 'dec1', 'dec2', 'mjd1', 'mjd2', 'date1', 'date2', 'fakeid1', 'fakeid2', 'id1', 'id2', \
                   'expnum1', 'expnum2', 'exptime1', 'exptime2', 'band1', 'band2', 'ccd1', 'ccd2', \
                   'mag1', 'mag2', 'ml_score1', 'ml_score2', 'd_para']
        return pd.DataFrame(data = pairs.T, columns = columns)
    
class link_pairs:
    def __init__(self, pair_df):
        self.df = pair_df
        self.dis_range = [100+10*i for i in range(100)]
        self.scan_range = 15
        self.cand_num = 0
        map(self.link, self.dis_range)
        
    def link(self, dis):
        print "checking distance: {}".format(dis)
        self.test_pairs = self.df[np.logical_and(self.df.d_para <= dis + self.scan_range, self.df.d_para >= dis - self.scan_range)]
        #print self.test_pairs
        print "n of pairs: {}".format(len(self.test_pairs))
        print "deparallaxing..."
        list1 = [list(self.test_pairs.ra1), list(self.test_pairs.dec1), list(self.test_pairs.mjd1), np.zeros(len(self.test_pairs))+dis]
        list2 = [list(self.test_pairs.ra2), list(self.test_pairs.dec2), list(self.test_pairs.mjd2), np.zeros(len(self.test_pairs))+dis]
        self.dpar_ra1, self.dpar_dec1 = topo_to_bary(list1)
        self.dpar_ra2, self.dpar_dec2 = topo_to_bary(list2)
        self.v_ra = (self.dpar_ra2 - self.dpar_ra1) / (self.test_pairs.mjd2 - self.test_pairs.mjd1)
        self.v_dec = (self.dpar_dec2 - self.dpar_dec1) / (self.test_pairs.mjd2 - self.test_pairs.mjd1)
        print "cutting in Velo..."
        #same_v = map(self.vcheck, self.v_ra, self.v_dec)
        print "line fitting..."
        same_line = map(self.check_line, self.dpar_ra1, self.dpar_ra2, self.dpar_dec1, self.dpar_dec2)
        #print test_pairs[same_line_v[0]]
        #print self.test_pairs.index
        #print len(self.test_pairs.index), len(same_line_v), len(same_line_v[0])
        print "checking orbit..."
        map(self.check_Orbit, self.test_pairs.index, same_line)
        
    def vcheck(self, test_v_ra, test_v_dec):
        ra_check = abs(np.array(self.v_ra) - test_v_ra) < 0.01/3600.
        dec_check = abs(np.array(self.v_dec) - test_v_dec) < 0.01/3600.
        return ra_check*dec_check
        
    def check_line(self, x1, x2, y1, y2):
        x_list = []
        y_list = []
        for i in np.array(zip(self.dpar_ra1, self.dpar_ra2, self.dpar_dec1, self.dpar_dec2)):
            x3, x4, y3, y4 = i
            x = [x1, x2, x3, x4]
            y = [y1, y2, y3, y4]
            x_list.append(x)
            y_list.append(y)
            
        lcheck = np.array(map(self.fit_line, x_list, y_list))
        return lcheck 
        
    def fit_line(self, x, y):
        result, res, rank, singular, rcond = np.polyfit(x, y, 1, full=True)
        return ((res[0]*(180/np.pi*3600)**2)/4.)**0.5 < 30 

    def check_Orbit(self, pair_index, mask, chisq_cut=2):
        pair = self.test_pairs.T[pair_index]
        #print pair_index
        #print pair
        #print "checking Orbit..."
        print pair.fakeid1, pair.fakeid2,
        x1 = pair.ra1 
        y1 = pair.dec1
        x2 = pair.ra2
        y2 = pair.dec2
        d1 = pair.mjd1 -15019.5
        d2 = pair.mjd2 -15019.5
        
        for i in self.test_pairs.index[mask]:
            x3 = self.test_pairs.T[i].ra1
            y3 = self.test_pairs.T[i].dec1
            x4 = self.test_pairs.T[i].ra2
            y4 = self.test_pairs.T[i].dec2
            d3 = self.test_pairs.T[i].mjd1 -15019.5
            d4 = self.test_pairs.T[i].mjd2 -15019.5
            ra = [x1, x2, x3, x4]
            dec = [y1, y2, y3, y4]
            date = [d1, d2, d3, d4]
            if len(np.unique(date)) == 4:
                ralist = [str(ephem.hours(j)) for j in ra]
                declist = [str(ephem.degrees(j)) for j in dec]
                datelist = [str(ephem.date(j)) for j in date]
                #print ralist
                #print declist
                #print datelist
                orbit = Orbit(dates=datelist, ra=ralist, dec=declist, obscode=np.ones(4, dtype=int)*807, err=0.25)
                if orbit.chisq/orbit.ndof<chisq_cut:
                    print self.test_pairs.T[i].fakeid1, self.test_pairs.T[i].fakeid2,
                    abg = orbit.get_elements_abg()[0]
                    with open("{0}cand_{1}.abg.json".format(output_dir, self.cand_num), 'w') as fp:
                        json.dump(abg, fp)
                    self.output(pair, self.test_pairs.T[i])
        print ' '

    def ra_to_str(self, ra):
        deg = ra*180/np.pi
        hh = int(deg/15.)
        mm = int((deg/15. - hh)*60)
        ss = ((deg/15. - hh)*60 - mm)*60
        return "{0}:{1:02}:{2:05.2f}".format(hh, mm, ss)
            
    def dec_to_str(self, dec):
        deg = dec*180/np.pi
        de = int(deg)
        mm = int((abs(deg) - abs(de))*60)
        ss = ((abs(deg) - abs(de))*60 - mm)*60
        return "{0}:{1:02}:{2:05.2f}".format(de, mm, ss)

    def output(self, pair1, pair2):
        #date,ra,dec,expnum,exptime,band,ccd,mag,ml_score,objid,fakeid 
        cand_id = [pair1['id1'], pair1['id2'], pair2['id1'], pair2['id2']]
        cand_ra = [pair1['ra1'], pair1['ra2'], pair2['ra1'], pair2['ra2']]
        cand_dec = [pair1['dec1'], pair1['dec2'], pair2['dec1'], pair2['dec2']]
        cand_date = [pair1['date1'], pair1['date2'], pair2['date1'], pair2['date2']]
        cand_expnum = [pair1['expnum1'], pair1['expnum2'], pair2['expnum1'], pair2['expnum2']]
        cand_exptime = [pair1['exptime1'], pair1['exptime2'], pair2['exptime1'], pair2['exptime2']]
        cand_band = [pair1['band1'], pair1['band2'], pair2['band1'], pair2['band2']]
        cand_ccd = [pair1['ccd1'], pair1['ccd2'], pair2['ccd1'], pair2['ccd2']]
        cand_mag = [pair1['mag1'], pair1['mag2'], pair2['mag1'], pair2['mag2']]
        cand_ml_score = [pair1['ml_score1'], pair1['ml_score2'], pair2['ml_score1'], pair2['ml_score2']]
        cand_fakeid = [pair1['fakeid1'], pair1['fakeid2'], pair2['fakeid1'], pair2['fakeid2']]
        cand_ra = map(self.ra_to_str, cand_ra)
        cand_dec = map(self.dec_to_str, cand_dec)
            
        new_cand = np.array([cand_date, cand_ra, cand_dec, cand_expnum, cand_exptime, cand_band, \
                             cand_ccd, cand_mag, cand_ml_score, cand_id, cand_fakeid])
        columns = ['date','ra','dec','expnum','exptime','band','ccd','mag','ml_score','objid','fakeid'] 
        candidate = pd.DataFrame(data = new_cand.T, columns = columns)
        candidate = candidate.drop_duplicates()
        candidate = candidate.sort_values(by='date')
       
        candidate.to_csv("{0}cand_{1}.csv".format(output_dir, self.cand_num), index=False)
        self.cand_num += 1              

                    
def main():
    season = sys.argv[1]
    global working_dir
    global output_dir
    working_dir = 'wsdiff_catalogs/season{}/fakesonly/'.format(season)
    #working_dir = '/nfs/lsa-spacerocks/wsdiff_catalogs/season{}/nofakes/'.format(season)
    output_dir = working_dir + '2pairs/'
    os.system('mkdir {}'.format(output_dir))
    linkmap_list  = glob.glob(working_dir+'wsdiff_season*Y?_griz_fakesonly_linkmap.pickle')
    #linkmap_list  = glob.glob(working_dir+'wsdiff_season*Y?_griz_nofakes_linkmap.pickle')
    det_file_list = [i.replace('_linkmap.pickle', '.csv') for i in linkmap_list]
    Pairs = pairs(linkmap_list, det_file_list)
    Pairs_100 = Pairs.df.drop(Pairs.df[Pairs.df.d_para < 100].index)
    del Pairs
    gc.collect()
    two_pairs = link_pairs(Pairs_100)
    
if __name__ == '__main__':
    main()