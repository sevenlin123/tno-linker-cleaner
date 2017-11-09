#!/usr/bin/env python

##########################################################################
#
# connect.py, version 0.1
#
# Connect triplets to the pairs in the other year. 
#
# Author: 
# Edward Lin: hsingwel@umich.edu
##########################################################################

from __future__ import division
import gc
import pandas as pd
import numpy as np
import ephem
from Orbit import Orbit
import pickle
from deparallax import topo_to_bary
from scipy.spatial import cKDTree as KDTree
from scipy.stats import linregress
import os, sys
pd.options.mode.chained_assignment = None

class deparallaxed_triplets:
    def __init__(self):
        self.triplets = None #pd.read_csv(working_dir+triplet_file)
        self.detections = None #all_detections_Y2
        #self.new_triplets()
        #self.useful_triplets()
    
    def get_tri_pos(self):  
        obj1 = pd.DataFrame(self.triplets['id1'])
        obj2 = pd.DataFrame(self.triplets['id2'])
        obj3 = pd.DataFrame(self.triplets['id3'])
        obj1 = obj1.join(self.detections.set_index('objid'), on='id1')
        obj2 = obj2.join(self.detections.set_index('objid'), on='id2')
        obj3 = obj3.join(self.detections.set_index('objid'), on='id3')
    
        ra1 = obj1['ra'].values*np.pi/180.
        ra2 = obj2['ra'].values*np.pi/180.
        ra3 = obj3['ra'].values*np.pi/180.

        dec1 = obj1['dec'].values*np.pi/180.
        dec2 = obj2['dec'].values*np.pi/180.
        dec3 = obj3['dec'].values*np.pi/180.
        
        mjd1 = obj1['mjd'].values
        mjd2 = obj2['mjd'].values
        mjd3 = obj3['mjd'].values
        
        expnum1 = obj1['expnum'].values
        expnum2 = obj2['expnum'].values
        expnum3 = obj3['expnum'].values
        
        exptime1 = obj1['exptime'].values
        exptime2 = obj2['exptime'].values
        exptime3 = obj3['exptime'].values
        
        band1 = obj1['band'].values
        band2 = obj2['band'].values
        band3 = obj3['band'].values
        
        ccd1 = obj1['ccd'].values
        ccd2 = obj2['ccd'].values
        ccd3 = obj3['ccd'].values
        
        mag1 = obj1['mag'].values
        mag2 = obj2['mag'].values
        mag3 = obj3['mag'].values
        
        ml_score1 = obj1['ml_score'].values
        ml_score2 = obj2['ml_score'].values
        ml_score3 = obj3['ml_score'].values
        
        fakeid1 = obj1['fakeid'].values
        fakeid2 = obj2['fakeid'].values
        fakeid3 = obj3['fakeid'].values
        
        date1 = obj1['date'].values
        date2 = obj2['date'].values
        date3 = obj3['date'].values
        
        dbary = self.triplets['dbary'].values
        return [ra1, ra2, ra3], [dec1, dec2, dec3], [mjd1, mjd2, mjd3], [dbary, dbary, dbary],\
               [expnum1, expnum2, expnum3], [exptime1, exptime2, exptime3], [band1, band2, band3],\
               [ccd1, ccd2, ccd3,], [mag1, mag2, mag3], [ml_score1, ml_score2, ml_score3],\
               [fakeid1, fakeid2, fakeid3], [date1, date2, date3]
    
    def dePara_tri(self, tri_pos_list):
        list0 = [tri_pos_list[0][0], tri_pos_list[1][0], tri_pos_list[2][0], tri_pos_list[3][0]]
        list1 = [tri_pos_list[0][1], tri_pos_list[1][1], tri_pos_list[2][1], tri_pos_list[3][1]]
        list2 = [tri_pos_list[0][2], tri_pos_list[1][2], tri_pos_list[2][2], tri_pos_list[3][2]]
        new_ra1, new_dec1 = topo_to_bary(list0)
        new_ra2, new_dec2 = topo_to_bary(list1)
        new_ra3, new_dec3 = topo_to_bary(list2)   
        ra1 = tri_pos_list[0][0]
        ra2 = tri_pos_list[0][1]
        ra3 = tri_pos_list[0][2]  
        dec1 = tri_pos_list[1][0]
        dec2 = tri_pos_list[1][1]
        dec3 = tri_pos_list[1][2] 
        mjd1 = tri_pos_list[2][0]
        mjd2 = tri_pos_list[2][1]
        mjd3 = tri_pos_list[2][2]   
        vRA12 = (new_ra2 - new_ra1) / (mjd2-mjd1) 
        vDEC12 = (new_dec2 - new_dec1) / (mjd2-mjd1)
        vRA23 = (new_ra3 - new_ra2) / (mjd3-mjd2) 
        vDEC23 = (new_dec3 - new_dec2) / (mjd3-mjd2)
        expnum1 = tri_pos_list[4][0]
        expnum2 = tri_pos_list[4][1]
        expnum3 = tri_pos_list[4][2]
        exptime1 = tri_pos_list[5][0]
        exptime2 = tri_pos_list[5][1]
        exptime3 = tri_pos_list[5][2]
        band1 = tri_pos_list[6][0]
        band2 = tri_pos_list[6][1]
        band3 = tri_pos_list[6][2]  
        ccd1 = tri_pos_list[7][0]
        ccd2 = tri_pos_list[7][1]
        ccd3 = tri_pos_list[7][2]
        mag1 = tri_pos_list[8][0]
        mag2 = tri_pos_list[8][1]
        mag3 = tri_pos_list[8][2]  
        ml1 = tri_pos_list[9][0]
        ml2 = tri_pos_list[9][1]
        ml3 = tri_pos_list[9][2]   
        fakeid1 = tri_pos_list[10][0]
        fakeid2 = tri_pos_list[10][1]
        fakeid3 = tri_pos_list[10][2]
        date1 = tri_pos_list[11][0]
        date2 = tri_pos_list[11][1]
        date3 = tri_pos_list[11][2]
        self.triplets = self.triplets.assign(new_ra1 = pd.Series(new_ra1))
        self.triplets = self.triplets.assign(new_ra2 = pd.Series(new_ra2))
        self.triplets = self.triplets.assign(new_ra3 = pd.Series(new_ra3))
        self.triplets = self.triplets.assign(new_dec1 = pd.Series(new_dec1))
        self.triplets = self.triplets.assign(new_dec2 = pd.Series(new_dec2))
        self.triplets = self.triplets.assign(new_dec3 = pd.Series(new_dec3))
        self.triplets = self.triplets.assign(new_vRA12 = pd.Series(vRA12))
        self.triplets = self.triplets.assign(new_vDEC12 = pd.Series(vDEC12))
        self.triplets = self.triplets.assign(new_vRA23 = pd.Series(vRA23))
        self.triplets = self.triplets.assign(new_vDEC23 = pd.Series(vDEC23))
        self.triplets = self.triplets.assign(ra1 = pd.Series(ra1))
        self.triplets = self.triplets.assign(ra2 = pd.Series(ra2))
        self.triplets = self.triplets.assign(ra3 = pd.Series(ra3))
        self.triplets = self.triplets.assign(dec1 = pd.Series(dec1))
        self.triplets = self.triplets.assign(dec2 = pd.Series(dec2))
        self.triplets = self.triplets.assign(dec3 = pd.Series(dec3))
        self.triplets = self.triplets.assign(mjd1 = pd.Series(mjd1))
        self.triplets = self.triplets.assign(mjd2 = pd.Series(mjd2))
        self.triplets = self.triplets.assign(mjd3 = pd.Series(mjd3))   
        self.triplets = self.triplets.assign(expnum1 = pd.Series(expnum1))
        self.triplets = self.triplets.assign(expnum2 = pd.Series(expnum2))
        self.triplets = self.triplets.assign(expnum3 = pd.Series(expnum3))
        self.triplets = self.triplets.assign(exptime1 = pd.Series(exptime1))
        self.triplets = self.triplets.assign(exptime2 = pd.Series(exptime2))
        self.triplets = self.triplets.assign(exptime3 = pd.Series(exptime3)) 
        self.triplets = self.triplets.assign(band1 = pd.Series(band1))
        self.triplets = self.triplets.assign(band2 = pd.Series(band2))
        self.triplets = self.triplets.assign(band3 = pd.Series(band3))
        self.triplets = self.triplets.assign(ccd1 = pd.Series(ccd1))
        self.triplets = self.triplets.assign(ccd2 = pd.Series(ccd2))
        self.triplets = self.triplets.assign(ccd3 = pd.Series(ccd3))
        self.triplets = self.triplets.assign(mag1 = pd.Series(mag1))
        self.triplets = self.triplets.assign(mag2 = pd.Series(mag2))
        self.triplets = self.triplets.assign(mag3 = pd.Series(mag3))
        self.triplets = self.triplets.assign(ml_score1 = pd.Series(ml1))
        self.triplets = self.triplets.assign(ml_score2 = pd.Series(ml2))
        self.triplets = self.triplets.assign(ml_score3 = pd.Series(ml3))
        self.triplets = self.triplets.assign(fakeid1 = pd.Series(fakeid1))
        self.triplets = self.triplets.assign(fakeid2 = pd.Series(fakeid2))
        self.triplets = self.triplets.assign(fakeid3 = pd.Series(fakeid3))
        self.triplets = self.triplets.assign(date1 = pd.Series(date1))
        self.triplets = self.triplets.assign(date2 = pd.Series(date2))
        self.triplets = self.triplets.assign(date3 = pd.Series(date3))

    def new_triplets(self):
        tri_pos_list = self.get_tri_pos()
        self.dePara_tri(tri_pos_list)

    def useful_triplets(self):
        self.useful_triplet = np.logical_and(abs(self.triplets['new_vDEC12']-self.triplets['new_vDEC23']) <= 0.00005, \
                                        abs(self.triplets['new_vRA12']-self.triplets['new_vRA23']) <= 0.00005)
                                        
class connecting_pairs:
    def __init__(self):
        self.triplets = None#triplets
        self.n_of_tri = None #len(triplets)
        self.pairs = None #pickle.load(open(working_dir+pair_file))
        self.detections = None #all_detections_Y3
        self.new_pairs = None
        self.should_be_detected = 0
        self.lost_in_useful_cut = 0
        self.lost_in_vcut = 0
        self.lost_in_linecheck = 0
        self.lost_in_orbfit = 0
        self.new_discoveries = 0
        self.cand_num = 0
        self.not_real = 0
        self.head150 = 0
        self.det150 = 0
        self.head801 = 0
        self.det801 = 0
        self.pair_list = []
        #self.gen_pair_list()
        self.pair_info0 = None #self.getDatePos()

    def getDatePos(self):
        obj1 = pd.DataFrame(data = np.array(self.pair_list).T[0], columns = ['id'])
        obj2 = pd.DataFrame(data = np.array(self.pair_list).T[1], columns = ['id'])
        obj1 = obj1.join(self.detections.set_index('objid'), on='id')
        obj2 = obj2.join(self.detections.set_index('objid'), on='id')
        
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
        
        return [ra1, ra2], [dec1, dec2], [mjd1, mjd2], [np.zeros(len(obj1)), np.zeros(len(obj2))],\
               [fakeid1, fakeid2], [id1, id2], [expnum1, expnum2], [exptime1, exptime2],\
               [band1, band2], [ccd1, ccd2], [mag1, mag2], [ml_score1, ml_score2], [date1, date2]

    def dePara_pair(self, pair_pos_list):
        del self.new_pairs
        gc.collect()
        list0 = [pair_pos_list[0][0], pair_pos_list[1][0], pair_pos_list[2][0], pair_pos_list[3][0]]
        list1 = [pair_pos_list[0][1], pair_pos_list[1][1], pair_pos_list[2][1], pair_pos_list[3][1]]
        new_ra1, new_dec1 = topo_to_bary(list0)
        new_ra2, new_dec2 = topo_to_bary(list1)             
        ra1 = pair_pos_list[0][0]
        ra2 = pair_pos_list[0][1]      
        dec1 = pair_pos_list[1][0]
        dec2 = pair_pos_list[1][1]        
        id1 = pair_pos_list[5][0]
        id2 = pair_pos_list[5][1]     
        mjd1 = pair_pos_list[2][0]
        mjd2 = pair_pos_list[2][1]     
        fakeid1 = pair_pos_list[4][0]
        fakeid2 = pair_pos_list[4][1] 
        
        expnum1 = pair_pos_list[6][0]
        expnum2 = pair_pos_list[6][1]
        
        exptime1 = pair_pos_list[7][0]
        exptime2 = pair_pos_list[7][1]
        
        band1 = pair_pos_list[8][0]
        band2 = pair_pos_list[8][1]
        
        ccd1 = pair_pos_list[9][0]
        ccd2 = pair_pos_list[9][1]
        
        mag1 = pair_pos_list[10][0]
        mag2 = pair_pos_list[10][1]
        
        ml_score1 = pair_pos_list[11][0]
        ml_score2 = pair_pos_list[11][1]
        
        date1 = pair_pos_list[12][0]
        date2 = pair_pos_list[12][1]
        
        vRA12 = (new_ra2 - new_ra1) / (mjd2-mjd1) 
        vDEC12 = (new_dec2 - new_dec1) / (mjd2-mjd1)
        
        new_pairs = np.array([id1, id2, new_ra1, new_ra2, new_dec1, new_dec2, vRA12, vDEC12,\
                              mjd1, mjd2, fakeid1, fakeid2, ra1, ra2, dec1, dec2, expnum1, expnum2, \
                              exptime1, exptime2, band1, band2, ccd1, ccd2, mag1, mag2, ml_score1, ml_score2,
                              date1, date2])
        columns = ['id1', 'id2', 'new_ra1', 'new_ra2', 'new_dec1', 'new_dec2', 'vRA12', 'vDEC12', \
                   'mjd1', 'mjd2', 'fakeid1', 'fakeid2', 'ra1', 'ra2', 'dec1', 'dec2', 'expnum1', 'expnum2',\
                   'exptime1', 'exptime2', 'band1', 'band2', 'ccd1', 'ccd2', 'mag1', 'mag2', 'ml_score1',\
                   'ml_score2', 'date1', 'date2'] 
        self.new_pairs = pd.DataFrame(data = new_pairs.T, columns = columns)
    
    def check_useful_pairs(self, tri_RA, tri_DEC, tri_mjd, dbary, vRA, vDEC):
        ra1 = self.pair_info0[0][0]
        ra2 = self.pair_info0[0][1]
        dec1 = self.pair_info0[1][0]
        dec2 = self.pair_info0[1][1]
        mjd1 = self.pair_info0[2][0]
        mjd2 = self.pair_info0[2][1]
        dbary1 = self.pair_info0[3][0] + dbary
        dbary2 = self.pair_info0[3][1] + dbary
        fakeid1 = self.pair_info0[4][0]
        fakeid2 = self.pair_info0[4][1]
        id1 = self.pair_info0[5][0]
        id2 = self.pair_info0[5][1]
        
        expnum1 = self.pair_info0[6][0]
        expnum2 = self.pair_info0[6][1]
        
        exptime1 = self.pair_info0[7][0]
        exptime2 = self.pair_info0[7][1]
        
        band1 = self.pair_info0[8][0]
        band2 = self.pair_info0[8][1]
        
        ccd1 = self.pair_info0[9][0]
        ccd2 = self.pair_info0[9][1]
        
        mag1 = self.pair_info0[10][0]
        mag2 = self.pair_info0[10][1]
        
        ml_score1 = self.pair_info0[11][0]
        ml_score2 = self.pair_info0[11][1]
        
        date1 = self.pair_info0[12][0]
        date2 = self.pair_info0[12][1]        
        
        predict_ra = tri_RA + vRA * (mjd1.mean()-tri_mjd)
        predict_dec = tri_DEC + vDEC * (mjd1.mean()-tri_mjd)
        distance =  ((predict_ra-ra1)**2 + (predict_dec-dec1)**2)**0.5
        mask = distance < 2.0 * np.pi/180.
        
        
        useful_pairs = ([ra1[mask], ra2[mask]], [dec1[mask], dec2[mask]], [mjd1[mask], mjd2[mask]], \
                        [dbary1[mask], dbary2[mask]], [fakeid1[mask], fakeid2[mask]], [id1[mask], id2[mask]], \
                        [expnum1[mask], expnum2[mask]], [exptime1[mask], exptime2[mask]], \
                        [band1[mask], band2[mask]], [ccd1[mask], ccd2[mask]], [mag1[mask], mag2[mask]],\
                        [ml_score1[mask], ml_score2[mask]], [date1[mask], date2[mask]]) 
        return useful_pairs
    
    def gen_pair_list(self):
        for i in self.pairs.keys():
            if self.pairs[i] != []:
                for j in self.pairs[i]:
                    self.pair_list.append([i,j])
    
    def check_line(self, x4, x5, y4, y5):
        x1 = self.test_tri['new_ra1']
        y1 = self.test_tri['new_dec1']
        x2 = self.test_tri['new_ra2']
        y2 = self.test_tri['new_dec2']
        x3 = self.test_tri['new_ra3']
        y3 = self.test_tri['new_dec3']
        #x4 = self.new_pairs.loc[match, ['new_ra1']].values[0]
        #y4 = self.new_pairs.loc[match, ['new_dec1']].values[0]
        #x5 = self.new_pairs.loc[match, ['new_ra2']].values[0]
        #y5 = self.new_pairs.loc[match, ['new_dec2']].values[0]
        result, res, rank, singular, rcond = np.polyfit([x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5], 1, full=True)
        return res[0]*180/np.pi*3600 < 0.1
    
    def check_Orbit(self, ra, dec, date, chisq_cut=2):
        #print "checking Orbit..."
        x1 = self.test_tri['ra1'] 
        y1 = self.test_tri['dec1']
        x2 = self.test_tri['ra2']
        y2 = self.test_tri['dec2']
        x3 = self.test_tri['ra3']
        y3 = self.test_tri['dec3']
        d1 = self.test_tri['mjd1'] -15019.5
        d2 = self.test_tri['mjd2'] -15019.5
        d3 = self.test_tri['mjd3'] -15019.5
        ra = [x1, x2, x3] + list(ra)
        dec = [y1, y2, y3] + list(dec)
        date = [d1, d2, d3] + list(date)
        #d4 = self.new_pairs.loc[match, ['mjd1']].values[0] -15019.5
        #d5 = self.new_pairs.loc[match, ['mjd2']].values[0] -15019.5
        ralist = [str(ephem.hours(i)) for i in ra]
        declist = [str(ephem.degrees(i)) for i in dec]
        datelist = [str(ephem.date(i)) for i in date]
        orbit = Orbit(dates=datelist, ra=ralist, dec=declist, obscode=np.ones(5, dtype=int)*807, err=0.15)
        return orbit.chisq/orbit.ndof<chisq_cut
                  
    def connect(self, index):
        if len(self.pair_list) == 0: 
            return 0 
        triplet = self.triplets.loc[index]
        self.test_tri = triplet
        self.fakeid = triplet['fakeid']
        if self.fakeid == -1:
            self.not_real += 1
        elif self.fakeid < 160000000:
            self.head150 += 1
        else:
            self.head801 += 1
        
        print "triplet number: {0}/{1}, fakeid: {2}".format(index, self.n_of_tri, self.fakeid)
 
        found_in_pairs0 = self.fakeid in self.pair_info0[4][0] and self.fakeid in self.pair_info0[4][1]
        if found_in_pairs0 and self.fakeid < 160000000:
            self.should_be_detected += 1
            self.det150 += 1
        elif found_in_pairs0 and self.fakeid > 160000000:
            self.should_be_detected += 1
            self.det801 += 1
    
        vRA = (triplet['new_ra3'] - triplet['new_ra1']) / (triplet['mjd3'] - triplet['mjd1'])
        vDEC = (triplet['new_dec3'] - triplet['new_dec1']) / (triplet['mjd3'] - triplet['mjd1'])
        
        pair_pos_list = self.check_useful_pairs(triplet['ra1'], triplet['dec1'], triplet['mjd1'], triplet['dbary'], vRA, vDEC)
        if found_in_pairs0 and len(pair_pos_list[0][0]) == 0:
            self.lost_in_useful_cut += 1
            return 0
        elif len(pair_pos_list[0][0]) == 0:
            return 0
            
        self.dePara_pair(pair_pos_list)
        #print "triplet fakeid: {0}".format(triplet['fakeid'])
        found_in_pairs = len(self.new_pairs.loc[np.logical_and(self.new_pairs['fakeid1'] == triplet['fakeid'], self.new_pairs['fakeid2'] == triplet['fakeid'])])
        #print "Number of pairs with correct fakeid: {0}".format(found_in_pairs)       
        if found_in_pairs == 0 and  found_in_pairs0 != 0:
            self.lost_in_useful_cut += 1
        pair_mjd1_list = np.array(self.new_pairs.groupby(['mjd1']).sum().index)
        pair_mjd2_list = np.array(self.new_pairs.groupby(['mjd2']).sum().index)
        tree_v = KDTree(zip(self.new_pairs['vRA12'].values, self.new_pairs['vDEC12'].values))
        predicted_ra1 = np.array(triplet['ra3'] + vRA*(pair_mjd1_list-triplet['mjd3']))
        predicted_dec1 = np.array(triplet['dec3'] + vDEC*(pair_mjd1_list-triplet['mjd3']))
        predicted_ra2 = np.array(triplet['ra3'] + vRA*(pair_mjd2_list-triplet['mjd3']))
        predicted_dec2 = np.array(triplet['dec3'] + vDEC*(pair_mjd2_list-triplet['mjd3']))
        predicted_ra1[predicted_ra1<0] += 2*np.pi 
        predicted_ra2[predicted_ra2<0] += 2*np.pi
        predicted_ra1[predicted_ra1 > 2*np.pi] -= 2*np.pi 
        predicted_ra2[predicted_ra2 > 2*np.pi] -= 2*np.pi 
        v_match = np.array(tree_v.query_ball_point([vRA, vDEC], 30/3600.* np.pi/180.))
        v_match.sort()
        found_in_pairs_vmatch = np.logical_and(self.new_pairs.loc[v_match]['fakeid1'] == self.fakeid, self.new_pairs.loc[v_match]['fakeid2'] == self.fakeid).sum()
        #print "Number of pairs after v_cut: {0}".format(len(v_match))
        if len(v_match) == 0 and found_in_pairs != 0:
            self.lost_in_vcut += 1
            return 0
        elif len(v_match) == 0:
            return 0
        elif found_in_pairs_vmatch ==0 and found_in_pairs != 0:
            self.lost_in_vcut += 1
                
        #self.slope, self.intercept, r_value, p_value, self.std = linregress(np.array(triplet[['ra1', 'ra2', 'ra3']]),\
        #                                                                    np.array(triplet[['dec1', 'dec2', 'dec3']]))
        #mask_vcheck = self.new_pairs.index.isin(match)     
        ra1 = self.new_pairs.loc[v_match]['new_ra1'].values
        ra2 = self.new_pairs.loc[v_match]['new_ra2'].values
        dec1 = self.new_pairs.loc[v_match]['new_dec1'].values
        dec2 = self.new_pairs.loc[v_match]['new_dec2'].values
        line_check = np.array(map(self.check_line, ra1, ra2, dec1, dec2))
        #print "Number of pairs after line_check: {0}".format(line_check.sum())
        found_after_linecheck = len(self.new_pairs.loc[v_match[line_check]][np.logical_and(self.new_pairs.loc[v_match[line_check]]['fakeid1'] == triplet['fakeid'], self.new_pairs.loc[v_match[line_check]]['fakeid2'] == triplet['fakeid'])])
        #print "Number of pairs with correct fakeid after line_check: {0}".format(found_after_linecheck)
        if line_check.sum() == 0 and found_in_pairs_vmatch != 0:
            self.lost_in_linecheck +=1
            return 0
        elif line_check.sum() == 0:
            return 0
        elif found_after_linecheck ==0 and found_in_pairs_vmatch != 0:
            self.lost_in_linecheck +=1
        
        self.remain_pairs0 = self.new_pairs.loc[v_match[line_check]]
        ra1 = self.new_pairs.loc[v_match[line_check]]['ra1'].values
        ra2 = self.new_pairs.loc[v_match[line_check]]['ra2'].values
        dec1 = self.new_pairs.loc[v_match[line_check]]['dec1'].values
        dec2 = self.new_pairs.loc[v_match[line_check]]['dec2'].values
        mjd1 = self.new_pairs.loc[v_match[line_check]]['mjd1'].values - 15019.5
        mjd2 = self.new_pairs.loc[v_match[line_check]]['mjd2'].values - 15019.5
        #print len(ra1)
        orbit_check = np.array(map(self.check_Orbit, zip(ra1, ra2), zip(dec1, dec2), zip(mjd1, mjd2)))
        self.remain_pairs = self.new_pairs.loc[v_match[line_check][orbit_check]]
        #print "Number of pairs after orbit_check: {0}".format(len(self.remain_pairs))
        if len(self.remain_pairs) == 0 and found_after_linecheck != 0:
            self.lost_in_orbfit +=1
            return 0
        elif len(self.remain_pairs) != 0 and found_after_linecheck == 0:
            self.new_discoveries += 1
            self.output(triplet, self.remain_pairs)
            return 1
        elif len(self.remain_pairs) == 0:
            return 0
        else:
            self.output(triplet, self.remain_pairs)
            return 1

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
        
    def output(self, triplet, pairs):
        #date,ra,dec,expnum,exptime,band,ccd,mag,ml_score,objid,fakeid
        cand_id = [triplet['id1'], triplet['id2'], triplet['id3']] + list(pairs['id1']) + list(pairs['id2'])
        cand_ra = [triplet['ra1'], triplet['ra2'], triplet['ra3']] + list(pairs['ra1']) + list(pairs['ra2'])
        cand_dec = [triplet['dec1'], triplet['dec2'], triplet['dec3']] + list(pairs['dec1']) + list(pairs['dec2'])
        cand_date = [triplet['date1'], triplet['date2'], triplet['date3']] + list(pairs['date1']) + list(pairs['date2'])
        cand_expnum = [triplet['expnum1'], triplet['expnum2'], triplet['expnum3']] + list(pairs['expnum1']) + list(pairs['expnum2'])
        cand_exptime = [triplet['exptime1'], triplet['exptime2'], triplet['exptime3']] + list(pairs['exptime1']) + list(pairs['exptime2'])
        cand_band = [triplet['band1'], triplet['band2'], triplet['band3']] + list(pairs['band1']) + list(pairs['band2'])
        cand_ccd = [triplet['ccd1'], triplet['ccd2'], triplet['ccd3']] + list(pairs['ccd1']) + list(pairs['ccd2'])
        cand_mag = [triplet['mag1'], triplet['mag2'], triplet['mag3']] + list(pairs['mag1']) + list(pairs['mag2'])
        cand_ml_score = [triplet['ml_score1'], triplet['ml_score2'], triplet['ml_score3']] + list(pairs['ml_score1']) + list(pairs['ml_score2'])
        cand_fakeid = [triplet['fakeid1'], triplet['fakeid2'], triplet['fakeid3']] + list(pairs['fakeid1']) + list(pairs['fakeid2'])
        cand_ra = map(self.ra_to_str, cand_ra)
        cand_dec = map(self.dec_to_str, cand_dec)
        
        new_cand = np.array([cand_date, cand_ra, cand_dec, cand_expnum, cand_exptime, cand_band, \
                              cand_ccd, cand_mag, cand_ml_score, cand_id, cand_fakeid])
        columns = ['date','ra','dec','expnum','exptime','band','ccd','mag','ml_score','objid','fakeid'] 
        candidate = pd.DataFrame(data = new_cand.T, columns = columns)
        candidate = candidate.drop_duplicates()
        candidate = candidate.sort_values(by='date')
        candidate.to_csv("{0}{1}cand_{2}.csv".format(working_dir, output_dir, self.cand_num), index=False)
        self.cand_num += 1
        
    def link_all(self):
        self.result = map(self.connect, list(self.triplets.index))
        print "starts with 150: {}".format(self.head150)
        print "starts with 801: {}".format(self.head801)
        print "not fake: {}".format(self.not_real)
        print "should be detected: {}".format(self.should_be_detected)
        print "should be detected 150: {}".format(self.det150)
        print "should be detected 801: {}".format(self.det801)
        print "detected: {}".format(np.array(self.result).sum())
        print "lost in useful_cut: {}".format(self.lost_in_useful_cut)
        print "lost in vcut: {}".format(self.lost_in_vcut)
        print "lost in linecheck: {}".format(self.lost_in_linecheck)
        print "lost in orbit: {}".format(self.lost_in_orbfit)
        print "new discoveries: {}".format(self.new_discoveries)
        
def main():
    chunk_id = sys.argv[1]
    season = sys.argv[2]
    global working_dir
    global output_dir
    working_dir = 'wsdiff_catalogs/season{}/fakesonly/'.format(season)
    #working_dir = '/nfs/lsa-spacerocks/wsdiff_catalogs/season{}/nofakes/'.format(season)
    triplet_file = 'CHUNK_Y2/chunk_{}/chunk_good_triplets.csv'.format(chunk_id)
    detections_file_Y2 = "wsdiff_season{}_Y2_griz_fakesonly.csv".format(season)
    detections_file_Y3 = "wsdiff_season{}_Y3_griz_fakesonly.csv".format(season)
    pair_file = 'wsdiff_season{}_Y3_griz_fakesonly_linkmap.pickle'.format(season)
    output_dir = "linker_cands_Y2/chunk_{}/".format(chunk_id)
    try:
        os.mkdir('{0}/{1}'.format(working_dir, 'linker_cands_Y2'))
    except OSError:
        pass
    try:
    	os.mkdir('{0}/{1}'.format(working_dir, output_dir))
    except OSError:
	pass
    all_detections_Y2 = pd.read_csv(working_dir+detections_file_Y2)
    all_detections_Y3 = pd.read_csv(working_dir+detections_file_Y3)
    tri = deparallaxed_triplets()
    tri.triplets = pd.read_csv(working_dir+triplet_file)
    tri.detections = all_detections_Y2
    tri.new_triplets()
    tri.useful_triplets()       

    conn = connecting_pairs()
    conn.triplets = tri.triplets[tri.useful_triplet]
    conn.n_of_tri = len(conn.triplets)
    conn.pairs = pickle.load(open(working_dir+pair_file))
    conn.detections = all_detections_Y3
    conn.gen_pair_list()
    conn.pair_info0 = conn.getDatePos()
    conn.link_all()

if __name__ == '__main__':
    main()
    
    
    
