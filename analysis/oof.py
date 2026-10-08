import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.argv=['x']; import importlib.util
spec=importlib.util.spec_from_file_location('r5','r5_analysis.py'); r5=importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('v2_analysis.csv').rename(columns={'class_name':'label','VV_VH_ratio':'VV_VH'})
le=LabelEncoder().fit(r5.CLASSES); y=le.transform(df.label); order=[list(le.classes_).index(c) for c in r5.CLASSES]
xk,yk,_=r5.project_km(df.longitude,df.latitude)
res={}; rng=np.random.default_rng(0)
for size in [None,1.0]:
  tag='random' if size is None else '1km'
  g=np.arange(len(df)) if size is None else r5.blocks(xk,yk,size)
  preds={}
  for rep in range(3):
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(df,y,g))
    for name,(mk,cols) in r5.make_models().items():
      p=np.empty(len(y),int)
      for a,b in folds:
        m=mk(); 
        if hasattr(m,'seed'): m.seed=rep
        p[b]=m.fit(df[cols].values[a],y[a]).predict(df[cols].values[b])
      preds.setdefault(name,[]).append(p)
  out={}
  for name,ps in preds.items():
    accs=[accuracy_score(y,p) for p in ps]; kap=[cohen_kappa_score(y,p) for p in ps]; f1=[f1_score(y,p,average='macro') for p in ps]
    p0=ps[0]; ci=r5.boot_ci(y,p0,rng)
    rec={c:float((p0[y==le.transform([c])[0]]==le.transform([c])[0]).mean()) for c in r5.CLASSES}
    out[name]=dict(acc=float(np.mean(accs)),acc_sd=float(np.std(accs)),kappa=float(np.mean(kap)),f1=float(np.mean(f1)),acc_ci=ci['acc_ci'],kappa_ci=ci['kappa_ci'],recall=rec)
    if name!=r5.REF: out[name]['mcnemar']=r5.mcnemar_exact(y,preds[r5.REF][0],p0)
  out['_cm_ref']=confusion_matrix(y,preds[r5.REF][0],labels=order).tolist()
  res[tag]=out
  print(f'\n== pooled out-of-fold, {tag} (n={len(y)}, 5-fold x 3 repeats)')
  for k,v in out.items():
    if k.startswith('_'): continue
    mc=v.get('mcnemar',{}).get('p')
    print(f"{k:42s} acc {v['acc']*100:5.1f}% ±{v['acc_sd']*100:.1f} (CI {v['acc_ci'][0]*100:.1f}-{v['acc_ci'][1]*100:.1f}) kappa {v['kappa']:.3f} F1 {v['f1']:.3f} rec R/V/B {v['recall']['Rice']:.2f}/{v['recall']['Vegetation']:.2f}/{v['recall']['Built-up']:.2f}  p {'' if mc is None else round(mc,4)}")
  print('CM ref (rows Rice,Veg,Built):', out['_cm_ref'])
json.dump(res,open('v2_out/oof_results.json','w'),indent=1)
