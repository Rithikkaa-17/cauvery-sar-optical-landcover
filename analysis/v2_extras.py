import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import importlib.util
spec=importlib.util.spec_from_file_location('r5','r5_analysis.py'); r5=importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('v2_out/v2_features_all.csv')
summ=[f'{p}_{s}' for p in ['VV','VH'] for s in ['mean','sd','min','max','range','doy_min']]
le=LabelEncoder().fit(r5.CLASSES); y=le.transform(df.class_name)
xk,yk,_=r5.project_km(df.longitude,df.latitude); g=r5.blocks(xk,yk,1.0)
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(df,y,g))
rng=np.random.default_rng(42); out={}
for tag,cols,mk in [('annual_DNN',['NDVI','VH','VV','VV_VH_ratio'],lambda: r5.DNN()),
                    ('summary_LR',summ+['NDVI'],lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000)))]:
    drops={c:[] for c in cols}
    for a,b in folds:
        m=mk().fit(df[cols].values[a],y[a]); Xb=df[cols].values[b]; base=accuracy_score(y[b],m.predict(Xb))
        for j,c in enumerate(cols):
            for _ in range(30):
                Xp=Xb.copy(); Xp[:,j]=rng.permutation(Xp[:,j]); drops[c].append(base-accuracy_score(y[b],m.predict(Xp)))
    out[tag]={c:(float(np.mean(v)),float(np.std(v))) for c,v in drops.items()}
    print(tag); [print(f'  {c:12s} {v[0]:.3f} ± {v[1]:.3f}') for c,v in sorted(out[tag].items(),key=lambda t:-t[1][0])]
json.dump(out,open('v2_out/perm_importance_block.json','w'),indent=1)
# training curves for annual DNN: 80/20 stratified split, 10% of train as validation
import tensorflow as tf
tr,te=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
X=df[['NDVI','VH','VV','VV_VH_ratio']].values; sc=StandardScaler().fit(X[tr]); Xs=sc.transform(X)
a,v=train_test_split(tr,test_size=0.1,stratify=y[tr],random_state=42)
tf.keras.utils.set_random_seed(42)
m=tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)),tf.keras.layers.Dense(128,activation='relu'),tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dropout(0.2),tf.keras.layers.Dense(3,activation='softmax')])
m.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
es=tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
h=m.fit(Xs[a],y[a],validation_data=(Xs[v],y[v]),epochs=100,batch_size=64,verbose=0,callbacks=[es])
H=h.history; best=int(np.argmin(H['val_loss']))+1
print('epochs run',len(H['loss']),'best epoch',best,'train acc %.3f val acc %.3f'%(H['accuracy'][best-1],H['val_accuracy'][best-1]))
json.dump({k:[float(x) for x in v] for k,v in H.items()}|{'best_epoch':best},open('v2_out/train_history.json','w'))
