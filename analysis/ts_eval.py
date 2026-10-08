import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import importlib.util
spec=importlib.util.spec_from_file_location('r5','r5_analysis.py'); r5=importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
T=pd.read_csv('ts2.csv'); V=pd.read_csv('v2_analysis.csv')
df=V.merge(T.drop(columns=['class','class_name','.geo','PLOTID'],errors='ignore'),on='point_id')
vv=sorted([c for c in T.columns if c.startswith('VV_')]); vh=sorted([c for c in T.columns if c.startswith('VH_')])
days=(pd.to_datetime([c[3:] for c in vv])-pd.Timestamp('2024-06-01')).days.values
for pol,cols in [('VV',vv),('VH',vh)]:
    A=df[cols].values
    df[f'{pol}_mean']=A.mean(1); df[f'{pol}_sd']=A.std(1); df[f'{pol}_min']=A.min(1); df[f'{pol}_max']=A.max(1)
    df[f'{pol}_range']=A.max(1)-A.min(1); df[f'{pol}_doy_min']=days[A.argmin(1)]
summ=[f'{p}_{s}' for p in ['VV','VH'] for s in ['mean','sd','min','max','range','doy_min']]
FS={'Annual 4 features':['NDVI','VH','VV','VV_VH_ratio'],
    'S1 time series (58)':vv+vh,
    'S1 time series + NDVI (59)':vv+vh+['NDVI'],
    'S1 summary (12) + NDVI':summ+['NDVI']}
MOD={'LogReg':lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)),
     'RF':lambda: RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=42),
     'DNN':lambda: r5.DNN()}
le=LabelEncoder().fit(r5.CLASSES); y=le.transform(df.class_name); order=[list(le.classes_).index(c) for c in r5.CLASSES]
xk,yk,_=r5.project_km(df.longitude,df.latitude); rng=np.random.default_rng(0)
REF=('Annual 4 features','DNN'); res={}
for tag,g in [('random',np.arange(len(df))),('1km',r5.blocks(xk,yk,1.0))]:
  preds={}
  for rep in range(3):
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(df,y,g))
    for fs,cols in FS.items():
      for mn,mk in MOD.items():
        p=np.empty(len(y),int)
        for a,b in folds:
          m=mk()
          if mn=='DNN': m.seed=rep
          p[b]=m.fit(df[cols].values[a],y[a]).predict(df[cols].values[b])
        preds.setdefault((fs,mn),[]).append(p)
  out={}
  best=None
  for k,ps in preds.items():
    p0=ps[0]; ci=r5.boot_ci(y,p0,rng)
    rec={c:float((p0[y==le.transform([c])[0]]==le.transform([c])[0]).mean()) for c in r5.CLASSES}
    out[' | '.join(k)]=dict(acc=float(np.mean([accuracy_score(y,p) for p in ps])),acc_sd=float(np.std([accuracy_score(y,p) for p in ps])),
      kappa=float(np.mean([cohen_kappa_score(y,p) for p in ps])),f1=float(np.mean([f1_score(y,p,average='macro') for p in ps])),
      acc_ci=ci['acc_ci'],recall=rec,mcnemar_vs_annualDNN=r5.mcnemar_exact(y,preds[REF][0],p0) if k!=REF else None,
      cm=confusion_matrix(y,p0,labels=order).tolist())
  res[tag]=out
  print(f'\n== {tag} (pooled OOF, n={len(y)}, 5-fold x3)')
  for k,v in out.items():
    mc=v['mcnemar_vs_annualDNN']
    print(f"{k:38s} acc {v['acc']*100:5.1f}% ±{v['acc_sd']*100:.1f} CI {v['acc_ci'][0]*100:.1f}-{v['acc_ci'][1]*100:.1f} k {v['kappa']:.3f} F1 {v['f1']:.3f} rec R/V/B {v['recall']['Rice']:.2f}/{v['recall']['Vegetation']:.2f}/{v['recall']['Built-up']:.2f} p {'' if mc is None else round(mc['p'],4)}")
json.dump(res,open('v2_out/ts_results.json','w'),indent=1)
df.to_csv('v2_out/v2_features_all.csv',index=False)
