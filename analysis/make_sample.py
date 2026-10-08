# Reproduces sample_points_1000_seed42.csv (simple random sample, min spacing 200 m)
import numpy as np, pandas as pd
rng=np.random.default_rng(42); lat0,lat1,lon0,lon1=10.85,11.15,79.05,79.45
kx=111.32*np.cos(np.deg2rad(11.0)); ky=110.57; pts=[]
while len(pts)<1000:
    la=rng.uniform(lat0,lat1); lo=rng.uniform(lon0,lon1)
    if all(np.hypot((lo-p[1])*kx,(la-p[0])*ky)>=0.2 for p in pts): pts.append((la,lo))
d=pd.DataFrame(pts,columns=['latitude','longitude']); d.insert(0,'point_id',[f'P{i:04d}' for i in range(1,1001)])
d.round(6).to_csv('sample_points_1000_seed42.csv',index=False)
