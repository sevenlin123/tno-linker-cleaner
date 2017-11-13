import pandas as pd
import numpy as np
import ephem
from Orbit import Orbit

def fit_orbit(df_obs):
    df_obs = df_obs.loc[['#' not in row['date'] for ind, row in df_obs.iterrows()]]   # filter comment lines
    nobs = len(df_obs)
    ralist = [ephem.hours(r) for r in df_obs['ra'].values]
    declist = [ephem.degrees(d) for d in df_obs['dec'].values]
    datelist = [ephem.date(d) for d in df_obs['date'].values]
    orbit = Orbit(dates=datelist, ra=ralist, dec=declist, obscode=np.ones(nobs, dtype=int)*807, err=0.15)
    fakeid = df_obs['fakeid'][3]
    return orbit

obs_list = open('cands.list').readlines()
for i in obs_list:
    i=i.strip()
    obs = pd.read_csv(i)
    orb = fit_orbit(obs)
    print i, orb.chisq/orb.ndof
    print orb.elements
