"""
leakage_audit.py — measure entity leakage under the existing random splits.
Self-contained: no er_core, no models, no GPU.
"""
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(os.environ.get("ER_PROJECT_ROOT", Path(__file__).resolve().parents[1])).expanduser()
DATA = str(PROJECT_ROOT / "data")

CFG = {
    "DBLP": dict(a=f"{DATA}/dblp/DBLP1.csv",
                 b=f"{DATA}/dblp/Scholar.csv",
                 m=f"{DATA}/dblp/DBLP-Scholar_perfectMapping.csv",
                 ida="idDBLP", idb="idScholar"),
    "ECOM": dict(a=f"{DATA}/abt_buy/Abt.csv",
                 b=f"{DATA}/abt_buy/Buy.csv",
                 m=f"{DATA}/abt_buy/abt_buy_perfectMapping.csv",
                 ida="idAbt", idb="idBuy"),
}

def clean(df):
    df.columns = [str(c).replace("\ufeff", "").replace("ï»¿", "")
                  .strip().strip('"').strip() for c in df.columns]
    return df

def load(path):
    return clean(pd.read_csv(path, dtype=str, encoding="latin-1"))

for name, cfg in CFG.items():
    df_a, df_b, m = load(cfg["a"]), load(cfg["b"]), load(cfg["m"])
    idcol_a = [c for c in df_a.columns if c.lower() == "id"][0]
    idcol_b = [c for c in df_b.columns if c.lower() == "id"][0]
    a_ids = df_a[idcol_a].fillna("").astype(str).str.strip().tolist()
    b_present = set(df_b[idcol_b].fillna("").astype(str).str.strip())

    m[cfg["ida"]] = m[cfg["ida"]].fillna("").astype(str).str.strip()
    m[cfg["idb"]] = m[cfg["idb"]].fillna("").astype(str).str.strip()

    truth, retained = defaultdict(set), {}
    for ida, idb in m[[cfg["ida"], cfg["idb"]]].itertuples(index=False, name=None):
        if idb not in b_present:
            continue
        truth[ida].add(idb)
        retained.setdefault(ida, idb)

    # ---- connected components of the bipartite ground-truth graph ----
    adj = defaultdict(set)
    for ida, bset in truth.items():
        for idb in bset:
            adj["a:" + ida].add("b:" + idb)
            adj["b:" + idb].add("a:" + ida)

    seen, comps = set(), []
    for node in adj:
        if node in seen:
            continue
        q, comp = deque([node]), []
        seen.add(node)
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(comp)

    multi = [c for c in comps if sum(1 for u in c if u.startswith("a:")) > 1]
    n_multi = sum(sum(1 for u in c if u.startswith("a:")) for c in multi)

    print(f"\n=== {name} ===")
    print(f"  |A| = {len(a_ids):,}   matched source items = {len(truth):,}")
    print(f"  components                      : {len(comps):,}")
    print(f"  components with >1 source item  : {len(multi):,}")
    print(f"  source items in those components: {n_multi:,} "
          f"({100*n_multi/len(truth):.1f}% of matched)")
    print(f"  largest component (nodes)       : {max(len(c) for c in comps)}")

    # ---- realised leakage under the actual 50/50 splits ----
    tot_leak = tot_loose = tot_matched = 0
    for seed in range(42, 52):
        tr, te = train_test_split(a_ids, test_size=0.50, random_state=seed)
        train_pos = {retained[i] for i in tr if i in retained}
        train_any = {b for i in tr for b in truth.get(i, ())}
        leak = loose = n_matched = 0
        for i in te:
            tset = truth.get(i, set())
            if not tset:
                continue
            n_matched += 1
            if tset & train_pos:
                leak += 1
            elif tset & train_any:
                loose += 1
        tot_leak += leak; tot_loose += loose; tot_matched += n_matched
        if seed == 42:
            print(f"  seed 42: {leak} leaked / {loose} weak / "
                  f"{n_matched} matched test items")

    print(f"  ACROSS 10 SEEDS: {tot_leak} of {tot_matched} matched test items "
          f"= {100*tot_leak/tot_matched:.2f}% (strict)")
    print(f"                   {tot_loose} more shared a non-positive partner "
          f"= {100*tot_loose/tot_matched:.2f}% (weak)")