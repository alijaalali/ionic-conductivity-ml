# Builds the master feature table for OBELiX ionic-conductivity modeling: a composition
# tier (Magpie elemental-property statistics + lattice/space-group/Z, all 599 entries) and
# a structure tier (Voronoi-polyhedron statistics + CrystalNN coordination number, pooled
# over all sites and over Li sites only, 321 entries with a resolved CIF). Target is
# log10(ionic conductivity in S/cm).
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.site import VoronoiFingerprint
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition

warnings.filterwarnings("ignore")

OBELIX_DATA = Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX/data")
OUT_DIR = Path("/glade/work/ajalali/Ionic-Conductivity-ML-project/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAGPIE = ElementProperty.from_preset("magpie")
VORO = VoronoiFingerprint()
CNN = CrystalNN()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_obelix():
    import sys
    sys.path.insert(0, str(Path("/glade/derecho/scratch/ajalali/obelix-data/OBELiX")))
    from obelix import OBELiX

    ob_full = OBELiX(str(OBELIX_DATA))
    ob_ordered = OBELiX(str(OBELIX_DATA)).round_partial()
    return ob_full, ob_ordered


def composition_features(comp_str):
    try:
        comp = Composition(comp_str)
        feats = MAGPIE.featurize(comp)
        return dict(zip(MAGPIE.feature_labels(), feats))
    except Exception:
        return {}


def structure_features(struct):
    """Returns a dict of pooled (mean/std) Voronoi + coordination-number
    features, computed over all sites and, separately, over Li sites only."""
    n = len(struct)
    voro_rows = []
    cn_rows = []
    li_indices = [i for i, sp in enumerate(struct.species) if str(sp) == "Li"]

    for i in range(n):
        try:
            voro_rows.append(VORO.featurize(struct, i))
        except Exception:
            voro_rows.append([np.nan] * len(VORO.feature_labels()))
        try:
            cn_rows.append(CNN.get_cn(struct, i))
        except Exception:
            cn_rows.append(np.nan)

    voro_arr = np.array(voro_rows, dtype=float)
    cn_arr = np.array(cn_rows, dtype=float)

    feats = {}
    voro_labels = [f"voro_{lbl}" for lbl in VORO.feature_labels()]

    def pool(arr_subset, cn_subset, prefix):
        out = {}
        if len(arr_subset) == 0:
            for lbl in voro_labels:
                out[f"{prefix}_{lbl}_mean"] = np.nan
                out[f"{prefix}_{lbl}_std"] = np.nan
            out[f"{prefix}_coord_num_mean"] = np.nan
            out[f"{prefix}_coord_num_std"] = np.nan
            return out
        for j, lbl in enumerate(voro_labels):
            out[f"{prefix}_{lbl}_mean"] = np.nanmean(arr_subset[:, j])
            out[f"{prefix}_{lbl}_std"] = np.nanstd(arr_subset[:, j])
        out[f"{prefix}_coord_num_mean"] = np.nanmean(cn_subset)
        out[f"{prefix}_coord_num_std"] = np.nanstd(cn_subset)
        return out

    feats.update(pool(voro_arr, cn_arr, "all"))
    if li_indices:
        feats.update(pool(voro_arr[li_indices], cn_arr[li_indices], "li"))
    else:
        feats.update(pool(np.zeros((0, len(voro_labels))), np.zeros(0), "li"))

    feats["n_sites"] = n
    feats["n_li_sites"] = len(li_indices)
    feats["cell_volume"] = struct.volume
    feats["density"] = struct.density
    return feats


def main():
    t0 = time.time()
    ob_full, ob_ordered = load_obelix()
    log(f"Loaded OBELiX: {len(ob_full)} total entries")

    rows = []
    n_struct_ok = 0
    n_struct_failed = 0

    for i in range(len(ob_full)):
        e = ob_full[i]
        entry_id = e["ID"]
        ic = e["Ionic conductivity (S cm-1)"]
        try:
            ic_val = float(ic)
        except (TypeError, ValueError):
            continue
        if ic_val <= 0:
            continue

        row = {
            "ID": entry_id,
            "Reduced_Composition": e["Reduced Composition"],
            "True_Composition": e["True Composition"],
            "Z": e["Z"],
            "log_IC": np.log10(ic_val),
            "Space_group": e["Space group"],
            "Space_group_number": e["Space group #"],
            "a": e["a"], "b": e["b"], "c": e["c"],
            "alpha": e["alpha"], "beta": e["beta"], "gamma": e["gamma"],
            "Family": e["Family"],
            "has_structure": False,
        }
        row.update(composition_features(e["True Composition"]))
        rows.append(row)

    log(f"Built composition-tier rows for {len(rows)} entries with valid IC")

    rows_by_id = {r["ID"]: r for r in rows}
    for i in range(len(ob_ordered)):
        e = ob_ordered[i]
        entry_id = e["ID"]
        if entry_id not in rows_by_id:
            continue
        struct = e.get("structure")
        if struct is None:
            continue
        t_entry = time.time()
        try:
            sfeats = structure_features(struct)
            rows_by_id[entry_id].update(sfeats)
            rows_by_id[entry_id]["has_structure"] = True
            n_struct_ok += 1
        except Exception as exc:
            n_struct_failed += 1
            log(f"  structure featurization failed for {entry_id}: {exc}")

        dt = time.time() - t_entry
        log(f"  [{i+1}/{len(ob_ordered)}] {entry_id} ({len(struct)} sites): {dt:.2f}s")

    log(f"Structure-tier features: {n_struct_ok} ok, {n_struct_failed} failed")

    df = pd.DataFrame(rows)

    # duplicate-measurement groups: same True Composition + Space group number
    dup_key = df["True_Composition"].astype(str) + "|" + df["Space_group_number"].astype(str)
    dup_counts = dup_key.value_counts()
    df["duplicate_group_id"] = dup_key
    df["duplicate_group_size"] = dup_key.map(dup_counts)

    df.to_parquet(OUT_DIR / "master_table.parquet")
    log(f"Wrote {len(df)} rows, {len(df.columns)} columns to "
        f"{OUT_DIR / 'master_table.parquet'}")
    log(f"Entries with structure: {df['has_structure'].sum()}")
    log(f"Entries in a duplicate-measurement group (size>1): "
        f"{(df['duplicate_group_size'] > 1).sum()}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
