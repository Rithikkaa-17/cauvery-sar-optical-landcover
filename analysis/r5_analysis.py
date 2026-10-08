#!/usr/bin/env python3
"""
Cauvery SAR-optical land-cover study - analyses requested by Reviewer 5.

Runs every experiment behind the new numbers in the revised manuscript, on the
same 19,999-sample corpus and the same splits, so the reported values can be
reproduced end to end.

  R5-1  Label-reliability check: draws a stratified, blind sample of points for
        independent visual interpretation on high-resolution imagery, and
        (in --score mode) computes per-class agreement with Wilson 95% CIs.
  R5-2  Spatial evaluation: 1 km (and 2 km / 5 km sensitivity) grid-block
        GroupKFold cross-validation and a block-held-out 80/20 test, plus the
        train-test nearest-neighbour distance for each split type.
  R5-3  Baselines on identical splits: majority class, multinomial logistic
        regression, Random Forest, and optical-only (NDVI) / SAR-only
        (VV, VH, VV/VH) versions of both the DNN and the RF.
        Accuracy, Cohen's kappa, macro-F1, bootstrap 95% CIs, and exact
        McNemar tests versus the full-feature DNN.
  R5-5  Confusion matrix and permutation importance (fixed n_repeats=30).

Usage
-----
  python r5_analysis.py run   --csv corpus.csv [--label-col label] [--out r5_out]
  python r5_analysis.py score --key r5_out/label_check_KEY.csv \
                              --interp interpreted.csv [--out r5_out]

The CSV needs the four features, the class label, and coordinates: either
'lon'/'lat' (or longitude/latitude) columns or a GEE '.geo' GeoJSON column.
Column names are auto-detected (case-insensitive); override with flags.
"""
import argparse, json, os, sys, math, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binomtest

SEED = 42
N_PERM = 30
N_BOOT = 1000
CLASSES = ["Rice", "Vegetation", "Built-up"]
FUSION = ["NDVI", "VH", "VV", "VV_VH"]
OPTICAL = ["NDVI"]
SAR = ["VH", "VV", "VV_VH"]

# ----------------------------------------------------------------- loading
ALIASES = {
    "NDVI": ["ndvi"],
    "VH": ["vh", "vh_db", "vh_mean", "mean_vh"],
    "VV": ["vv", "vv_db", "vv_mean", "mean_vv"],
    "VV_VH": ["vv_vh", "vv/vh", "vv_vh_ratio", "ratio", "vvvh", "vv_vh_ratio_db", "pol_ratio"],
    "lon": ["lon", "longitude", "x", "long"],
    "lat": ["lat", "latitude", "y"],
    "label": ["label", "class", "landcover", "class_name", "lc", "target"],
}

def find_col(df, key, override=None):
    if override:
        return override
    low = {c.lower().strip(): c for c in df.columns}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    return None

def load(args):
    df = pd.read_csv(args.csv)
    ren = {}
    for k in ["NDVI", "VH", "VV", "VV_VH", "label"]:
        c = find_col(df, k, getattr(args, k.lower() + "_col", None))
        if c is None:
            sys.exit(f"Could not find column for {k}. Columns: {list(df.columns)}")
        ren[c] = k
    lon, lat = find_col(df, "lon", args.lon_col), find_col(df, "lat", args.lat_col)
    if lon and lat:
        ren[lon], ren[lat] = "lon", "lat"
        df = df.rename(columns=ren)
    elif ".geo" in df.columns:
        df = df.rename(columns=ren)
        g = df[".geo"].apply(json.loads)
        df["lon"] = g.apply(lambda d: d["coordinates"][0])
        df["lat"] = g.apply(lambda d: d["coordinates"][1])
    else:
        sys.exit("No coordinates found (need lon/lat or .geo). Spatial evaluation is impossible without them.")
    df = df.dropna(subset=FUSION + ["label", "lon", "lat"]).reset_index(drop=True)
    # map numeric labels to names if needed
    if pd.api.types.is_numeric_dtype(df["label"]):
        m = json.loads(args.label_map) if args.label_map else {0: "Rice", 1: "Vegetation", 2: "Built-up"}
        df["label"] = df["label"].map({int(k): v for k, v in m.items()})
    return df

def project_km(lon, lat):
    """Local equirectangular projection to km (the grid used for all reported results).
    Deliberately does not use pyproj/UTM, so that 1 km blocks are identical on every machine."""
    lat0 = np.deg2rad(lat.mean())
    return lon.values * 111.320 * np.cos(lat0), lat.values * 110.574, "local equirectangular"

def blocks(xk, yk, size_km):
    return (np.floor(xk / size_km).astype(int) * 100000 + np.floor(yk / size_km).astype(int))

# ----------------------------------------------------------------- models
def keras_available():
    try:
        import tensorflow  # noqa
        return True
    except Exception:
        return False

class DNN:
    """Dense(128)-BN-Dropout(0.3)-Dense(64)-Dropout(0.2)-Softmax(3), Adam, sparse CE,
    early stopping on a 10% stratified validation split of the training partition."""
    def __init__(self, seed=SEED):
        self.seed = seed
    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        if keras_available():
            import tensorflow as tf
            tf.keras.utils.set_random_seed(self.seed)
            m = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(3, activation="softmax")])
            m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.1, stratify=y, random_state=self.seed)
            es = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=100, batch_size=64, verbose=0, callbacks=[es])
            self.m, self.kind = m, "keras"
        else:
            from sklearn.neural_network import MLPClassifier
            self.m = MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=10, max_iter=300, random_state=self.seed).fit(Xs, y)
            self.kind = "sklearn-MLP (TensorFlow not installed)"
        return self
    def predict(self, X):
        Xs = self.sc.transform(X)
        if self.kind == "keras":
            return self.m.predict(Xs, verbose=0).argmax(1)
        return self.m.predict(Xs)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

def make_models():
    return {
        "Majority class":            (lambda: DummyClassifier(strategy="most_frequent"), FUSION),
        "Logistic regression (all 4)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), FUSION),
        "Random Forest (all 4)":     (lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED), FUSION),
        "Random Forest (optical-only: NDVI)": (lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED), OPTICAL),
        "Random Forest (SAR-only: VV, VH, VV/VH)": (lambda: RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED), SAR),
        "DNN (optical-only: NDVI)":  (lambda: DNN(), OPTICAL),
        "DNN (SAR-only: VV, VH, VV/VH)": (lambda: DNN(), SAR),
        "DNN (all 4, SAR+optical)":  (lambda: DNN(), FUSION),
    }
REF = "DNN (all 4, SAR+optical)"

# ----------------------------------------------------------------- stats
def metrics(y, p):
    return dict(acc=accuracy_score(y, p), kappa=cohen_kappa_score(y, p), f1=f1_score(y, p, average="macro"))

def boot_ci(y, p, rng):
    n = len(y); a, k, f = [], [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        a.append(accuracy_score(y[i], p[i])); k.append(cohen_kappa_score(y[i], p[i]))
        f.append(f1_score(y[i], p[i], average="macro"))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(acc_ci=q(a), kappa_ci=q(k), f1_ci=q(f))

def mcnemar_exact(y, p_ref, p_other):
    a = (p_ref == y); b = (p_other == y)
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return dict(ref_only=n01, other_only=n10, p=1.0)
    return dict(ref_only=n01, other_only=n10, p=float(binomtest(n01, n01 + n10, 0.5).pvalue))

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def nn_dist_km(xy_tr, xy_te):
    d, _ = NearestNeighbors(n_neighbors=1).fit(xy_tr).kneighbors(xy_te)
    return float(np.median(d)), float(np.percentile(d, 5))

def perm_importance(model, X, y, cols, rng):
    base = accuracy_score(y, model.predict(X)); out = {}
    for j, c in enumerate(cols):
        drops = []
        for _ in range(N_PERM):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - accuracy_score(y, model.predict(Xp)))
        out[c] = (float(np.mean(drops)), float(np.std(drops)))
    return out

# ----------------------------------------------------------------- run
def run(args):
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load(args)
    le = LabelEncoder().fit(CLASSES)  # alphabetical: Built-up, Rice, Vegetation
    order = [list(le.classes_).index(c) for c in CLASSES]
    y = le.transform(df["label"].values)
    xk, yk, crs = project_km(df["lon"], df["lat"])
    XY = np.c_[xk, yk]
    R = dict(n=len(df), class_counts=df["label"].value_counts().to_dict(), crs=crs,
             dnn_backend="keras" if keras_available() else "sklearn-MLP", seed=SEED, n_perm=N_PERM)
    print(f"Loaded {len(df)} samples; class counts {R['class_counts']}; projection {crs}; DNN backend {R['dnn_backend']}")

    # --- A. stratified random 80/20 (same protocol as Section 3.5)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    R["random"] = dict(n_train=len(tr), n_test=len(te), nn_km=nn_dist_km(XY[tr], XY[te]), models={})
    preds = {}
    for name, (mk, cols) in make_models().items():
        m = mk().fit(df[cols].values[tr], y[tr]); p = m.predict(df[cols].values[te]); preds[name] = p
        r = metrics(y[te], p); r.update(boot_ci(y[te], p, rng)); R["random"]["models"][name] = r
        print(f"[random] {name:45s} acc={r['acc']:.3f} kappa={r['kappa']:.3f} F1={r['f1']:.3f}")
        if name == REF:
            ref_model = m
    for name in preds:
        if name != REF:
            R["random"]["models"][name]["mcnemar_vs_ref"] = mcnemar_exact(y[te], preds[REF], preds[name])
    cm = confusion_matrix(y[te], preds[REF], labels=order)
    R["random"]["confusion_matrix_rows_true_cols_pred"] = dict(classes=CLASSES, matrix=cm.tolist())
    R["random"]["per_class"] = classification_report(y[te], preds[REF], labels=order, target_names=CLASSES, output_dict=True)
    R["perm_importance_random_test"] = perm_importance(ref_model, df[FUSION].values[te], y[te], FUSION, rng)

    # --- B. random stratified 5-fold CV (reference for the block CV)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    R["random_cv5"] = {}
    for name in [REF, "Random Forest (all 4)", "Logistic regression (all 4)"]:
        mk, cols = make_models()[name]; acc, kap = [], []
        for a, b in skf.split(idx, y):
            p = mk().fit(df[cols].values[a], y[a]).predict(df[cols].values[b])
            acc.append(accuracy_score(y[b], p)); kap.append(cohen_kappa_score(y[b], p))
        R["random_cv5"][name] = dict(acc_mean=float(np.mean(acc)), acc_sd=float(np.std(acc, ddof=1)),
                                     kappa_mean=float(np.mean(kap)), kappa_sd=float(np.std(kap, ddof=1)), folds=acc)
        print(f"[random CV5] {name:40s} acc={np.mean(acc):.3f}±{np.std(acc, ddof=1):.3f}")

    # --- C. spatial grid blocks
    R["spatial"] = {}
    sizes = [] if args.no_spatial else [1.0, 2.0, 5.0]
    if args.group_col and not args.no_spatial:
        sizes = ["field"] + sizes
    for size in sizes:
        g = df[args.group_col].values if size == "field" else blocks(xk, yk, size)
        S = dict(block_km=size, n_blocks=int(len(np.unique(g))),
                 median_samples_per_block=float(pd.Series(g).value_counts().median()), models={})
        # C1 block-held-out 80/20
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        a, b = next(gss.split(idx, y, g))
        S["holdout"] = dict(n_train=len(a), n_test=len(b), nn_km=nn_dist_km(XY[a], XY[b]),
                            test_class_counts=pd.Series(le.inverse_transform(y[b])).value_counts().to_dict())
        # C2 5-fold stratified group CV
        sgk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        folds = list(sgk.split(idx, y, g))
        names = list(make_models().keys()) if size in (1.0, "field") else [REF, "Random Forest (all 4)"]
        hp = {}
        for name in names:
            mk, cols = make_models()[name]
            p = mk().fit(df[cols].values[a], y[a]).predict(df[cols].values[b]); hp[name] = p
            h = metrics(y[b], p); h.update(boot_ci(y[b], p, rng))
            acc, kap, f1 = [], [], []
            for fa, fb in folds:
                pf = mk().fit(df[cols].values[fa], y[fa]).predict(df[cols].values[fb])
                acc.append(accuracy_score(y[fb], pf)); kap.append(cohen_kappa_score(y[fb], pf)); f1.append(f1_score(y[fb], pf, average="macro"))
            S["models"][name] = dict(holdout=h, cv5=dict(acc_mean=float(np.mean(acc)), acc_sd=float(np.std(acc, ddof=1)),
                                     kappa_mean=float(np.mean(kap)), kappa_sd=float(np.std(kap, ddof=1)),
                                     f1_mean=float(np.mean(f1)), folds=acc))
            print(f"[{size}] {name:40s} holdout acc={h['acc']:.3f}  CV5 acc={np.mean(acc):.3f}±{np.std(acc, ddof=1):.3f}")
        for name in hp:
            if name != REF:
                S["models"][name]["holdout"]["mcnemar_vs_ref"] = mcnemar_exact(y[b], hp[REF], hp[name])
        if size == 1.0:
            S["confusion_matrix_holdout_ref"] = confusion_matrix(y[b], hp[REF], labels=order).tolist()
        R["spatial"]["field" if size == "field" else f"{size:g}km"] = S

    # --- D. blind label-check sample (R5-1)
    per = args.check_per_class if not args.no_spatial else 0
    dfi = df.assign(_i=idx)
    samp = pd.concat([dfi[dfi["label"] == c].sample(n=min(per, int((dfi["label"] == c).sum())), random_state=SEED)
                      for c in CLASSES])
    samp = samp.sample(frac=1, random_state=SEED).reset_index(drop=True)
    samp["check_id"] = [f"LC{i:04d}" for i in range(1, len(samp) + 1)]
    samp[["check_id", "_i", "lon", "lat", "label"]].rename(columns={"_i": "row_index"}).to_csv(
        os.path.join(args.out, "label_check_KEY.csv"), index=False)
    blind = samp[["check_id", "lon", "lat"]].copy()
    blind["interpreted_class"] = ""   # Rice / Vegetation / Built-up / Other / Unclear
    blind["imagery_source_and_date"] = ""
    blind["interpreter"] = ""
    blind["confidence_1to3"] = ""
    blind.to_csv(os.path.join(args.out, "label_check_BLIND.csv"), index=False)
    with open(os.path.join(args.out, "label_check_BLIND.kml"), "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>')
        for _, r in blind.iterrows():
            f.write(f"<Placemark><name>{r.check_id}</name><Point><coordinates>{r.lon},{r.lat},0</coordinates></Point></Placemark>")
        f.write("</Document></kml>")
    R["label_check_sample"] = dict(per_class=per, n=len(samp))

    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(os.path.join(args.out, "r5_results.json"), "w") as f:
        json.dump(R, f, indent=2, default=conv)
    write_tables(R, args.out)
    print(f"\nDone. See {args.out}/r5_results.json and {args.out}/r5_tables.md")

def _cm_perm(R, L):
    L.append("\n## Confusion matrix, random split, DNN (rows true, cols predicted: Rice, Vegetation, Built-up)\n")
    L.append(str(R["random"]["confusion_matrix_rows_true_cols_pred"]["matrix"]))
    L.append(f"\n## Permutation importance (n_repeats = {R['n_perm']}, random-split test set)\n")
    for k, (m, s) in sorted(R["perm_importance_random_test"].items(), key=lambda t: -t[1][0]):
        L.append(f"- {k}: {m:.3f} ± {s:.3f}")
    L.append("\n## Random 5-fold CV\n")
    for k, v in R["random_cv5"].items():
        L.append(f"- {k}: {v['acc_mean']*100:.1f}% ± {v['acc_sd']*100:.1f} (kappa {v['kappa_mean']:.3f})")

def write_tables(R, out):
    L = ["# Reviewer 5 analysis results\n", f"n = {R['n']}; DNN backend: {R['dnn_backend']}; projection: {R['crs']}\n"]
    L.append("## Table 4 - Random stratified 80/20 split (test n = %d)\n" % R["random"]["n_test"])
    L.append("| Model | Accuracy (95% CI) | Kappa (95% CI) | Macro-F1 | McNemar vs DNN (p) |\n|---|---|---|---|---|")
    for k, v in R["random"]["models"].items():
        mc = v.get("mcnemar_vs_ref", {}).get("p", None)
        L.append(f"| {k} | {v['acc']*100:.1f}% ({v['acc_ci'][0]*100:.1f}-{v['acc_ci'][1]*100:.1f}) | "
                 f"{v['kappa']:.3f} ({v['kappa_ci'][0]:.3f}-{v['kappa_ci'][1]:.3f}) | {v['f1']:.3f} | "
                 f"{'-' if mc is None else ('<0.001' if mc < 0.001 else f'{mc:.3f}')} |")
    if not R["spatial"]:
        L.append("\n(spatial evaluation skipped)\n")
        _cm_perm(R, L); open(os.path.join(out, "r5_tables.md"), "w").write("\n".join(L) + "\n"); return
    S = R["spatial"]["1km"]
    L.append(f"\n## Table 4 (cont.) - 1 km block-held-out split (test n = {S['holdout']['n_test']}, blocks = {S['n_blocks']})\n")
    L.append("| Model | Hold-out accuracy (95% CI) | Hold-out kappa | 5-fold block CV accuracy (mean ± SD) |\n|---|---|---|---|")
    for k, v in S["models"].items():
        h, c = v["holdout"], v["cv5"]
        L.append(f"| {k} | {h['acc']*100:.1f}% ({h['acc_ci'][0]*100:.1f}-{h['acc_ci'][1]*100:.1f}) | {h['kappa']:.3f} | "
                 f"{c['acc_mean']*100:.1f}% ± {c['acc_sd']*100:.1f} |")
    if "field" in R["spatial"]:
        Fd = R["spatial"]["field"]
        L.append(f"\n## Field-held-out (groups = polygons; test n = {Fd['holdout']['n_test']}, fields = {Fd['n_blocks']})\n")
        L.append("| Model | Hold-out accuracy (95% CI) | Hold-out kappa | 5-fold field CV accuracy |\n|---|---|---|---|")
        for k, v in Fd["models"].items():
            h, c = v["holdout"], v["cv5"]
            L.append(f"| {k} | {h['acc']*100:.1f}% ({h['acc_ci'][0]*100:.1f}-{h['acc_ci'][1]*100:.1f}) | {h['kappa']:.3f} | {c['acc_mean']*100:.1f}% ± {c['acc_sd']*100:.1f} |")
    L.append("\n## Block-size sensitivity (DNN all 4)\n| Block | n blocks | Hold-out acc | CV5 acc | median train-test NN distance (km) |\n|---|---|---|---|---|")
    L.append(f"| random split | - | {R['random']['models'][REF]['acc']*100:.1f}% | {R['random_cv5'][REF]['acc_mean']*100:.1f}% ± {R['random_cv5'][REF]['acc_sd']*100:.1f} | {R['random']['nn_km'][0]:.3f} |")
    for s, v in R["spatial"].items():
        if s == "field":
            continue
        m = v["models"][REF]
        L.append(f"| {s} | {v['n_blocks']} | {m['holdout']['acc']*100:.1f}% | {m['cv5']['acc_mean']*100:.1f}% ± {m['cv5']['acc_sd']*100:.1f} | {v['holdout']['nn_km'][0]:.3f} |")
    L.append("\n## Confusion matrix, random split, DNN (rows true, cols predicted: Rice, Vegetation, Built-up)\n")
    L.append(str(R["random"]["confusion_matrix_rows_true_cols_pred"]["matrix"]))
    L.append(f"\n## Permutation importance (n_repeats = {R['n_perm']}, random-split test set)\n")
    for k, (m, s) in sorted(R["perm_importance_random_test"].items(), key=lambda t: -t[1][0]):
        L.append(f"- {k}: {m:.3f} ± {s:.3f}")
    open(os.path.join(out, "r5_tables.md"), "w").write("\n".join(L) + "\n")

# ----------------------------------------------------------------- score label check
def score(args):
    key = pd.read_csv(args.key); it = pd.read_csv(args.interp)
    d = key.merge(it[["check_id", "interpreted_class"]], on="check_id")
    d["interpreted_class"] = d["interpreted_class"].astype(str).str.strip()
    d = d[d["interpreted_class"].str.len() > 0]
    unclear = d["interpreted_class"].str.lower().isin(["unclear", "nan", ""])
    dd = d[~unclear]
    out = dict(n_interpreted=len(d), n_unclear=int(unclear.sum()), per_class={})
    k = int((dd.label == dd.interpreted_class).sum()); n = len(dd)
    out["overall"] = dict(agree=k, n=n, rate=k / n if n else None, wilson95=wilson(k, n))
    for c in CLASSES:
        s = dd[dd.label == c]; kc = int((s.interpreted_class == c).sum())
        out["per_class"][c] = dict(agree=kc, n=len(s), rate=kc / len(s) if len(s) else None, wilson95=wilson(kc, len(s)),
                                   interpreted_as=s.interpreted_class.value_counts().to_dict())
    cats = CLASSES + ["Other"]
    out["crosstab_label_rows_interpreted_cols"] = pd.crosstab(dd.label, dd.interpreted_class).reindex(index=CLASSES).fillna(0).astype(int).to_dict()
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(os.path.join(args.out, "label_check_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--csv", required=True); r.add_argument("--out", default="r5_out")
    for k in ["ndvi", "vh", "vv", "vv_vh", "label", "lon", "lat"]:
        r.add_argument(f"--{k}-col", dest=f"{k}_col", default=None)
    r.add_argument("--label-map", default=None, help='JSON e.g. {"0":"Rice","1":"Vegetation","2":"Built-up"}')
    r.add_argument("--check-per-class", type=int, default=100)
    r.add_argument("--group-col", default=None, help="field/polygon id column (e.g. poly_id) for a field-held-out evaluation")
    r.add_argument("--no-spatial", action="store_true", help="skip block evaluation and label-check sample (use when coordinates are not the true sample locations)")
    s = sp.add_parser("score")
    s.add_argument("--key", required=True); s.add_argument("--interp", required=True); s.add_argument("--out", default="r5_out")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)
