#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import argparse
import pandas as pd
import numpy as np

try:
    import pulp as pl
except ImportError:
    pl = None

def canon_seq_str(x):
    if isinstance(x, (list, tuple)):
        return "-".join(str(int(i)) for i in x) if x else "∅"
    s = str(x).strip()
    if not s or s in ("None", "∅"):
        return "∅"
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip() for p in s.replace(",", "-").split("-") if p.strip()]
    out = []
    for q in parts:
        try:
            out.append(str(int(float(q))))
        except:
            out.append(q)
    return "-".join(out) if out else "∅"

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

def load_top3_for_section(sec_dir: Path):
    xlsx = sec_dir / "Results.xlsx"
    if not xlsx.exists():
        return None, "Results.xlsx not found"

    try:
        df = pd.read_excel(xlsx, sheet_name="Summary")
    except Exception as e:
        return None, f"Cannot read Summary: {e}"

    if not has_cols(df, ["Seq","Area","NPV_Total"]):
        return None, "Summary missing Seq/Area/NPV_Total"

    df = df.copy()
    df["Seq_str"] = df["Seq"].map(canon_seq_str)

    # Coerce numeric
    for c in ["Area","NPV_Total","NPV_Agency","NPV_User","Agency_Cost_Abs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop do-nothing and any non-numeric Area/NPV_Total
    df = df[(df["Seq_str"]!="∅") & df["Area"].notna() & df["NPV_Total"].notna()].copy()
    if df.empty:
        return None, "No eligible alternatives (only ∅ or non-numeric metrics)"

    df = df.sort_values(["Area","NPV_Total"], ascending=[False, False]).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df)+1)
    top3 = df.head(3).copy()

    keep = [c for c in (
        "Section","Rank","Seq_str","Area","NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs"
    ) if c in top3.columns]
    top3.insert(0, "Section", sec_dir.name)
    return top3[["Section"] + keep], None

def compute_cost_per_lkm(sec_dir: Path, seq_str: str) -> float:
    xlsx = sec_dir / "Results.xlsx"
    try:
        dfc = pd.read_excel(xlsx, sheet_name="PerYearCosts")
    except Exception:
        return float("nan")
    if "Agency_PV" not in dfc.columns:
        return float("nan")
    dfc = dfc.copy()
    if "Seq" in dfc.columns:
        dfc["Seq_str"] = dfc["Seq"].map(canon_seq_str)
        dfc = dfc[dfc["Seq_str"] == seq_str]
    return float(dfc["Agency_PV"].sum())

def aggregate_top3(outputs_root: Path):
    print(f"Scanning outputs root: {outputs_root.resolve()}")
    coverage_rows, top3_frames = [], []

    if not outputs_root.exists():
        raise SystemExit(f"Outputs folder not found: {outputs_root}")

    for sec_dir in sorted(outputs_root.iterdir()):
        if not sec_dir.is_dir():
            continue
        sec_name = sec_dir.name
        xlsx = sec_dir / "Results.xlsx"
        if not xlsx.exists():
            coverage_rows.append({"Section": sec_name, "FoundSummary": False,
                                  "EligibleAltCount": 0, "Included": False,
                                  "Reason": "Results.xlsx not found"})
            continue
        try:
            df = pd.read_excel(xlsx, sheet_name="Summary")
        except Exception as e:
            coverage_rows.append({"Section": sec_name, "FoundSummary": True,
                                  "EligibleAltCount": 0, "Included": False,
                                  "Reason": f"Cannot read Summary: {e}"})
            continue

        # basic checks
        needed = {"Seq","Area","NPV_Total"}
        if not needed.issubset(df.columns):
            coverage_rows.append({"Section": sec_name, "FoundSummary": True,
                                  "EligibleAltCount": 0, "Included": False,
                                  "Reason": "Summary missing Seq/Area/NPV_Total"})
            continue

        # normalize sequence + numerics
        df = df.copy()
        df["Seq_str"] = df["Seq"].map(canon_seq_str)
        for c in ["Area","NPV_Total","NPV_Agency","NPV_User","Agency_Cost_Abs"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # (optional) warn if any non-DN rows lost due to NaNs in Area/NPV_Total
        bad = df[(df["Seq_str"] != "∅") & (df["Area"].isna() | df["NPV_Total"].isna())]
        if len(bad):
            print(f"[WARN] {sec_name}: dropped {len(bad)} non-DN rows with NaN Area/NPV_Total")

        dn = df[df["Seq_str"] == "∅"].head(1).copy()
        if dn.empty:
            dn = pd.DataFrame([{"Section": sec_name, "Seq_str": "∅",
                                "Area": 0.0, "NPV_Agency": 0.0, "NPV_User": 0.0,
                                "NPV_Total": 0.0, "Agency_Cost_Abs": 0.0, "Rank": 0}])
        else:
            dn.insert(0, "Section", sec_name)
            dn["Rank"] = 0
            for c in ["NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs"]:
                dn[c] = 0.0
            dn["Area"] = pd.to_numeric(dn.get("Area", 0.0), errors="coerce").fillna(0.0)
        
        # ---- choose Top-3 among non-DN by NPV first (ties → smaller Area)
        non_dn = df[(df["Seq_str"]!="∅") & df["Area"].notna() & df["NPV_Total"].notna()].copy()
        if non_dn.empty:
            t3 = dn
            eligible_count = 0
        else:
            non_dn = non_dn.sort_values(["NPV_Total","Area"], ascending=[False, False]).reset_index(drop=True)
            non_dn["Rank"] = np.arange(1, len(non_dn)+1)
            t3 = pd.concat([dn, non_dn.head(3)], ignore_index=True)
            eligible_count = int(non_dn["Rank"].max())
            
        # keep common columns and add Section
        keep = [c for c in ("Rank","Seq_str","Area","NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs") if c in t3.columns]
        
        if "Section" in t3.columns:
            t3 = t3.drop(columns=["Section"])
        
        t3.insert(0, "Section", sec_name)
        t3 = t3[["Section"] + keep]

        top3_frames.append(t3)
        coverage_rows.append({"Section": sec_name, "FoundSummary": True,
                              "EligibleAltCount": eligible_count, "Included": True, "Reason": ""})

    if not top3_frames:
        raise SystemExit("No valid sections found under outputs/. Did you generate Results.xlsx files?")

    agg = pd.concat(top3_frames, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows).sort_values("Section").reset_index(drop=True)

    # clean Section labels
    agg["Section"] = agg["Section"].astype(str)
    agg = agg[agg["Section"].str.lower() != "nan"]

    # helpful counts on contributing (non-DN) ranks
    per_sec_counts = agg[agg["Rank"] > 0].groupby("Section")["Rank"].max().rename("MaxRank").reset_index()
    n_ge1 = int((per_sec_counts["MaxRank"] >= 1).sum())
    n_ge2 = int((per_sec_counts["MaxRank"] >= 2).sum())
    n_ge3 = int((per_sec_counts["MaxRank"] >= 3).sum())
    n_secs = agg["Section"].nunique()
    print(f"Sections with ≥1 eligible alt: {n_ge1} | ≥2: {n_ge2} | ≥3: {n_ge3} | total contributing: {n_secs}")

    # rank-wise sums **exclude DN (Rank 0)**
    cols = [c for c in ["NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs"] if c in agg.columns]
    if cols:
        sums = agg[agg["Rank"] > 0].groupby("Rank")[cols].sum().reset_index()
        sums_meta = pd.DataFrame({
            "Metric": ["Num_Sections_Contributing","Num_With_≥1","Num_With_≥2","Num_With_≥3"],
            "Value":  [n_secs, n_ge1, n_ge2, n_ge3]
        })
    else:
        sums = pd.DataFrame()
        sums_meta = pd.DataFrame({"Metric": [], "Value": []})

    return agg, coverage, sums, sums_meta

def optimize_with_budget(agg: pd.DataFrame, outputs_root: Path, budget: float, mode: str):
    if pl is None:
        raise SystemExit("PuLP is not installed. Install with: pip install pulp")

    # Ensure Seq_str, numeric NPV
    if "Seq_str" not in agg.columns:
        if "Seq" in agg.columns:
            agg = agg.copy()
            agg["Seq_str"] = agg["Seq"].map(canon_seq_str)
        else:
            raise SystemExit("Input table lacks Seq/Seq_str columns.")
    agg = agg.copy()
    agg["NPV_Total"] = pd.to_numeric(agg["NPV_Total"], errors="coerce").fillna(0.0)

    # ensure section labels are strings and not NaN
    agg["Section"] = agg["Section"].astype(str)
    agg = agg[agg["Section"].str.lower() != "nan"]

    # Decide cost column 
    if mode == "abs":
        cost_col = "Agency_Cost_Abs"
    elif mode == "perLKM":
        cost_col = "Agency_Cost_perLKM"
    else:  # auto
        cost_col = "Agency_Cost_Abs" if "Agency_Cost_Abs" in agg.columns and not agg["Agency_Cost_Abs"].isna().all() else "Agency_Cost_perLKM"

    # If perLKM, compute from PerYearCosts (seq-specific)
    if cost_col == "Agency_Cost_perLKM":
        costs = []
        for _, r in agg.iterrows():
            c = compute_cost_per_lkm(outputs_root / r.Section, r.Seq_str)
            costs.append(c)
        agg["Agency_Cost_perLKM"] = costs
        print("Note: Budget is interpreted in $ per lane-km.")

    if cost_col not in agg.columns:
        raise SystemExit(f"Cost column {cost_col} missing after processing.")
    
    agg[cost_col] = pd.to_numeric(agg[cost_col], errors="coerce")
    
    # If ABS but some costs are NaN, try to compute a fallback from PerYearCosts
    if cost_col == "Agency_Cost_Abs" and agg[cost_col].isna().any():
        print("[INFO] Some Agency_Cost_Abs are NaN → computing perLKM fallbacks from PerYearCosts")
        fallback = []
        for _, r in agg.iterrows():
            if pd.isna(r["Agency_Cost_Abs"]):
                c = compute_cost_per_lkm(outputs_root / r.Section, r.Seq_str)
            else:
                c = r["Agency_Cost_Abs"]
            fallback.append(c)
        agg[cost_col] = fallback
    
    # After fallback, if still NaN, drop those rows (don’t pretend they cost 0)
    missing = agg[cost_col].isna()
    if missing.any():
        dropped = int(missing.sum())
        print(f"[WARN] Dropping {dropped} rows with unknown cost in {cost_col}")
        agg = agg[~missing].copy()

    # ---- Guarantee DN available per section (Rank 0, cost 0, NPV 0) ----
    rows = []
    for sec in sorted(agg["Section"].unique()):
        sub = agg[agg["Section"] == sec]
        has_dn = (sub["Seq_str"] == "∅").any()
        if not has_dn:
            rows.append({
                "Section": sec, "Rank": 0, "Seq_str": "∅", "Area": 0.0,
                "NPV_Agency": 0.0, "NPV_User": 0.0, "NPV_Total": 0.0,
                "Agency_Cost_Abs": 0.0, "Agency_Cost_perLKM": 0.0
            })
    if rows:
        agg = pd.concat([agg, pd.DataFrame(rows)], ignore_index=True)

    contributors = agg["Section"].nunique()

    # Make absolutely sure DN rows are zeroed in the optimization table
    dn_mask = agg["Seq_str"] == "∅"
    for c in ["NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs","Agency_Cost_perLKM"]:
        if c in agg.columns:
            agg.loc[dn_mask, c] = 0.0

    # ---- MILP: pick exactly one per section; maximize NPV; obey budget ----
    model = pl.LpProblem("NetworkSelection", pl.LpMaximize)
    def _ir(v): 
        try: return int(v)
        except: return 0
    keys = [(r.Section, _ir(r.Rank), r.Seq_str) for _, r in agg.iterrows()]
    x = pl.LpVariable.dicts("x", keys, lowBound=0, upBound=1, cat="Binary")

    npv = {(r.Section, _ir(r.Rank), r.Seq_str): float(r.NPV_Total) for _, r in agg.iterrows()}
    cost = {(r.Section, _ir(r.Rank), r.Seq_str): float(r[cost_col]) for _, r in agg.iterrows()}
    model += pl.lpSum(x[k] * npv[k] for k in keys)

    for sec in sorted(agg["Section"].unique(), key=str):
        model += pl.lpSum(x[k] for k in keys if k[0] == sec) == 1

    model += pl.lpSum(x[k] * cost[k] for k in keys) <= float(budget)

    model.solve(pl.PULP_CBC_CMD(msg=False))
    status = pl.LpStatus[model.status]
    print(f"Solver status: {status}")

    chosen = []
    for _, r in agg.iterrows():
        k = (r.Section, _ir(r.Rank), r.Seq_str)
        if pl.value(x[k]) and pl.value(x[k]) > 0.5:
            chosen.append(r)
    sel = pd.DataFrame(chosen)
    if sel.empty:
        raise SystemExit("No feasible solution under the given budget.")

    total_npv = float(sel["NPV_Total"].sum())
    total_cost = float(sel[cost_col].sum())
    tag = "abs" if cost_col == "Agency_Cost_Abs" else "perLKM"
    print(f"Contributing sections: {contributors}")
    print(f"Optimization selected {sel['Section'].nunique()} sections (should equal contributors).")
    print(f"Total NPV = {total_npv:,.2f}")
    print(f"Total Agency Cost ({tag}) = {total_cost:,.2f}")

    return sel, tag

def main():
    ap = argparse.ArgumentParser(description="Aggregate Top-3 per section and optimize under a budget.")
    ap.add_argument("--outputs", type=str, default="outputs", help="Path to outputs/ folder")
    ap.add_argument("--budget", type=float, required=True, help="Budget (ABS $ or $/lane-km depending on mode)")
    ap.add_argument("--mode", choices=["auto","abs","perLKM"], default="auto",
                    help="Budget interpretation: auto (default), abs, or perLKM")
    args = ap.parse_args()

    outputs_root = Path(args.outputs)

    # 1) Aggregate
    agg, coverage, sums, sums_meta = aggregate_top3(outputs_root)

    # 2) Optimize
    sel, tag = optimize_with_budget(agg, outputs_root, args.budget, args.mode)

    # 3) Write one Excel with everything
    out_xlsx = outputs_root / "_network_summary.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
        agg.to_excel(w, sheet_name="All_Top3", index=False)
        coverage.to_excel(w, sheet_name="Coverage", index=False)
        if not sums.empty:
            sums.to_excel(w, sheet_name="Rank_Sums", index=False)
            sums_meta.to_excel(w, sheet_name="Rank_Sums_Meta", index=False)
        sel.to_excel(w, sheet_name="Optimization", index=False)
    print(f"Wrote {out_xlsx}")

if __name__ == "__main__":
    main()

