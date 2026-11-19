"""Read-only measurement for debate 1: does a per-prompt 'each model once' constraint help on the existing holdout?
Uses artifacts/baseline/validation_pipeline.joblib (fit on train holdout only) and split.csv. Writes nothing to the repo."""
import json, math, sys
import numpy as np, pandas as pd, joblib
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

repo = sys.argv[1]
train = pd.read_csv(f'{repo}/train.csv', dtype=str, keep_default_na=False)
split = pd.read_csv(f'{repo}/artifacts/baseline/split.csv', dtype=str, keep_default_na=False)
print('split columns', list(split.columns))
role_col = [c for c in split.columns if c not in ('id', 'group', 'prompt_id', 'component')][0]
val_ids = set(split.loc[split[role_col].str.contains('valid'), 'id'])
val = train[train.id.isin(val_ids)].reset_index(drop=True)
pipe = joblib.load(f'{repo}/artifacts/baseline/validation_pipeline.joblib')
classes = list(pipe.classes_)
P = pipe.predict_proba(val.text.tolist())
logP = np.log(np.clip(P, 1e-12, 1))
y = val.model.to_numpy()
base = np.array(classes)[P.argmax(1)]
out = {'n_val': len(val), 'n_prompts': val.prompt_id.nunique(),
       'argmax_acc': float((base == y).mean()), 'argmax_bal': float(balanced_accuracy_score(y, base))}

def family(m):
    return m.split('-')[0].replace('qwen3.8', 'qwen').replace('glm', 'glm')
err = base != y
out['errors'] = int(err.sum())
out['errors_same_family'] = int(sum(family(a) == family(b) for a, b in zip(y[err], base[err])))

def assign(groups):
    pred = np.empty(len(val), dtype=object)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        reps = math.ceil(len(idx) / len(classes))
        cost = -np.tile(logP[idx], (1, reps))  # columns: classes repeated, each slot used once
        r, c = linear_sum_assignment(cost)
        pred[idx[r]] = np.array(classes)[c % len(classes)]
    return pred

oracle = assign(val.prompt_id.to_numpy())
out['oracle_prompt_hungarian_acc'] = float((oracle == y).mean())

# Prompt reconstruction without prompt_id: word TF-IDF topic clustering on validation texts only.
X = TfidfVectorizer(analyzer='word', stop_words='english', min_df=2, sublinear_tf=True).fit_transform(val.text)
k = val.prompt_id.nunique()
lab = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average').fit_predict(X.toarray())
out['cluster_ARI_vs_prompt'] = float(adjusted_rand_score(val.prompt_id, lab))
sizes = np.bincount(lab)
out['cluster_sizes_min_median_max'] = [int(sizes.min()), int(np.median(sizes)), int(sizes.max())]
clustered = assign(lab)
out['clustered_hungarian_acc'] = float((clustered == y).mean())
print(json.dumps(out, indent=1))
