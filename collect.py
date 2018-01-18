##########################################################################
#
# collect.py, version 0.1.1
#
# group the tracks from same objects with DBSCAN algorithm and orbit_fit
#
# Author: 
# Edward Lin: hsingwel@umich.edu
#
# v0.1.1: read info from json file; added some comments
##########################################################################
from __future__ import division
import os, glob, sys
import pandas as pd
import numpy as np
import ephem
from Orbit import Orbit
from sklearn.cluster import DBSCAN
import json

sys.setrecursionlimit(50000) 

# This is the object to collect tracks from the same minor planets
class collect_object:
    def __init__(self, obj_list):
        self.obj_list = obj_list
        self.df_obs_list = None
        self.orbits = None
        self.xyz = None
        self.abg = None
        self.tree_xyz = None
        self.read_obj()
        self.get_orbit()
        self.get_xyz()
        self.get_abg()
        self.new_tracks_clustering = None
        self.good_tracks = []
        self.clustering()
        self.group_tracks()
        self.clean_track()
        self.output()
    
    #read the track list
    def read_obj(self):
        self.df_obs_list = map(pd.read_csv, self.obj_list)
    
    #calculate the orbit of track
    def fit_orbit(self, df_obs):
        df_obs = df_obs.loc[['#' not in row['date'] for ind, row in df_obs.iterrows()]]   # filter comment lines
        nobs = len(df_obs)
        ralist = [ephem.hours(r) for r in df_obs['ra'].values]
        declist = [ephem.degrees(d) for d in df_obs['dec'].values]
        datelist = [ephem.date(d) for d in df_obs['date'].values]
        orbit = Orbit(dates=datelist, ra=ralist, dec=declist, obscode=np.ones(nobs, dtype=int)*807, err=0.15)
        fakeid = df_obs['fakeid'][3]
        return orbit, (fakeid-150000000)

    #get orbit for all of the tracks
    def get_orbit(self):
        print "get tracks' orbits..."
        orb = map(self.fit_orbit, self.df_obs_list)
        self.orbits, self.fakeids = zip(*orb)
    #read the xyz position of track
    def get_xyz(self):
        self.xyz = np.array([[i.xBary*i.barycentric_distance()[0], i.yBary*i.barycentric_distance()[0],\
                             i.zBary*i.barycentric_distance()[0]] for i in self.orbits])
    
    #read the abg phase space position of track    
    def get_abg(self):
        self.abg = np.array([[i.get_elements_abg()[0]['a'], i.get_elements_abg()[0]['b'], i.get_elements_abg()[0]['g'], \
                              i.get_elements_abg()[0]['adot'], i.get_elements_abg()[0]['bdot'], i.get_elements_abg()[0]['gdot']]\
                             for i in self.orbits])
    
    #label the tracks with similar abg phase space position
    def clustering(self):
        print "cluster the similar tracks..."
        db = DBSCAN(eps=0.00005, min_samples=1, n_jobs=-1)
        db.fit(self.abg)
        self.labels = db.labels_
        
    #group tracks with same label to one master track
    def group(self, label):
        m = 0
        for n, i in enumerate(self.labels):
            if i == label:
                m+=1
                try:
                    new_track = pd.concat([new_track, self.df_obs_list[n]], ignore_index=False)
                except NameError:
                    new_track = self.df_obs_list[n]
        return new_track.drop_duplicates().sort_values(by='date')
    
    #group every tracks
    def group_tracks(self):
        self.new_tracks_clustering = map(self.group, np.unique(self.labels))
    
    #test if the master tracks can be further linking with each other
    def group_tracks_orbfit(self, remain_tracks):
        print len(remain_tracks)
        if len(remain_tracks) >1:
            test_track = remain_tracks.pop(0)
            #print test_track
            tracks_to_keep = []
            fakeid = test_track.fakeid.mean()
            print fakeid
            for i in remain_tracks:
                orbit, _ = self.fit_orbit(pd.concat([test_track, i]).drop_duplicates().sort_values(by='date'))
                if orbit.chisq/orbit.ndof < 3.:
                    test_track = pd.concat([test_track, i]).drop_duplicates().sort_values(by='date')
                else:
                    tracks_to_keep.append(i)
            self.good_tracks.append(test_track)
            self.group_tracks_orbfit(tracks_to_keep)
        elif len(remain_tracks) ==1:
            self.good_tracks.append(remain_tracks[0])
            return 0
        else:
            return 0

    def clean_track(self):
        print "cleanup..."
        self.group_tracks_orbfit(self.new_tracks_clustering)
    
    # output every independent tracks    
    def output(self):
        for n, i in enumerate(self.good_tracks):
            i.to_csv("{0}/2pairs_cands/good_{1}.csv".format(working_dir, n), index=False)
            
def main():
    season = sys.argv[1]
    config_file = json.loads(open('config_test.json').read())
    global working_dir
    global output_dir
    working_dir = '{0}/season{1}/{2}/'.format(config_file['subdir'], season, config_file['sample'])
    #working_dir = './wsdiff_catalogs/season{0}/fakesonly/'.format(season)
    #working_dir = '/nfs/lsa-spacerocks/wsdiff_catalogs/season{0}/scrambled_fakes/'.format(season)
    os.system('mkdir {}/2pairs_cands'.format(working_dir))
    #obj_list = glob.glob("{}/2pairs/*.csv".format(working_dir))
    obj_list = glob.glob("{}/cands_Y?/good*.csv".format(working_dir)) 
    objects = collect_object(obj_list)

if __name__ == '__main__':
    main()
                
