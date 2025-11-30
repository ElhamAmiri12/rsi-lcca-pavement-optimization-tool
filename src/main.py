#!/usr/bin/env python
# coding: utf-8

# In[4]:


from __future__ import annotations
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox
import pandas as pd
import csv
from tkinter import filedialog as fd
import math, logging, os
from pathlib import Path
from typing import List, Sequence
import contextlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from copy import deepcopy
import numpy as np
import re, sys, unicodedata
import argparse, sys

# ======  helpers =========================
def _norm_key(s: str) -> str:
    """Normalize CSV header titles so variants/typos still match."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    # unify glyphs/variants
    s = (s.replace("–", "-").replace("—", "-").replace("−", "-"))
    # tolerate common variants/typos
    s = s.replace("truck %", "truck per").replace("% trucks", "truck per").replace("truck share", "truck per")
    # compress spaces and drop punctuation (keep alnum+space)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

TITLES_TO_KEYS = {
    _norm_key("Section name"): "Section_Name",
    _norm_key("Base type"): "S1",
    _norm_key("Pavement structure"): "S2",
    _norm_key("Subgrade type"): "S3",
    _norm_key("Functional class"): "S4",
    _norm_key("Analysis period"): "Design_life",

    _norm_key("ESAL"): "ESAL",
    _norm_key("SN"): "SN",
    _norm_key("Asphalt Thickness"): "ATK",
    _norm_key("FTC"): "FT",
    _norm_key("FI"): "FI",
    _norm_key("CI"): "CI",
    _norm_key("Precipitation"): "RAIN",
    _norm_key("Age at initial IRI"): "AGE_IRI",
    _norm_key("Initial IRI"): "INI_IRI",
    _norm_key("Length"): "LEN",
    _norm_key("Number of lanes"): "LANES",

    _norm_key("Coefficient of Asphalt"): "CO_A",
    _norm_key("Coefficient of Base"): "CO_B",
    _norm_key("Coefficient of Subbase"): "CO_SUB",

    _norm_key("Base thickness"): "TH_B",
    _norm_key("Subbase thickness"): "TH_SUB",
    _norm_key("Drainage coefficient base"): "DCO_B",
    _norm_key("Drainage coefficient of subbase"): "DCO_SUB",

    _norm_key("Overlay thickness"): "TH_OVER",
    _norm_key("Structural layer coefficient"): "CO_ASTL",

    _norm_key("Initial cost"): "Construction_Cost",
    _norm_key("Main cost"): "Maintenance_Cost",
    _norm_key("Rehab cost"): "Rehabilitation_Cost",
    _norm_key("Recon cost"): "Reconstruction_Cost",
    _norm_key("Interest"): "Interest_rate",
    _norm_key("Inflation"): "Inflation_rate",

    _norm_key("AADT"): "UC_AADT",
    _norm_key("Truck per"): "UC_TRUCK_SHARE",
    _norm_key("Free flow speed"): "UC_FFS_KPH",
    _norm_key("Minimum speed"): "UC_MIN_SPEED",
    _norm_key("IRI TTC"): "UC_IRI0",
    _norm_key("k"): "UC_K_KPH_PER_IRI",
    _norm_key("VOC car"): "UC_VOC_CAR",
    _norm_key("VOC truck"): "UC_VOC_TRK",
    _norm_key("VOT car"): "UC_VOT_CAR",
    _norm_key("VOT truck"): "UC_VOT_TRK",
    _norm_key("Occupancy car"): "UC_OCC_CAR",
    _norm_key("Occupancy truck"): "UC_OCC_TRK",
}

CSV_TITLES_FRIENDLY = [
    "Section name","Base type","Pavement structure","Subgrade type","Functional class","Analysis period",
    "ESAL","SN","Asphalt Thickness","FTC","FI","CI","Precipitation","Age at initial IRI","Initial IRI","Length","Number of lanes",
    "Coefficient of Asphalt","Coefficient of Base","Coefficient of Subbase",
    "Base thickness","Subbase thickness","Drainage coefficient base","Drainage coefficient of subbase",
    "Overlay thickness","Structural layer coefficient",
    "Initial cost","Main cost","Rehab cost","Recon cost","Interest","Inflation",
    "AADT","Truck per","Free flow speed","Minimum speed","IRI TTC","k",
    "VOC car","VOC truck","VOT car","VOT truck","Occupancy car","Occupancy truck",
]

# ---- CLI + mode flags -------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch", metavar="CSV", help="Process all sections from this CSV (no UI).")
parser.add_argument("--outroot", default="outputs", help="Root folder for outputs.")
args, _ = parser.parse_known_args()

HEADLESS = bool(args.batch)
BATCH_CSV = args.batch
OUTPUT_ROOT = Path(args.outroot)

def safe_folder_name(s: str) -> str:
    s = (s or "").strip()
    return "".join(ch if ch.isalnum() or ch in " _-." else "_" for ch in s) or "section"

a_IRI_Fine=[[0,-0.714,-0.759],[-0.713,0,-0.753],[0,-0.658,-0.702],[0,-0.674,-0.718],[-0.658,0,-0.698],[0,-0.734,-0.779]]
a_IRI_coarse=[[0,-0.77,-0.815],[-0.769,0,-0.81],[0,-0.714,-0.759],[0,-0.73,-0.774],[-0.714,0,-0.755],[0,-0.79,-0.835]]
a_IRI_rock=[[0,-0.671,-0.716],[-0.67,0,-0.711],[0,-0.615,-0.66],[0,-0.631,-0.676],[-0.615,0,-0.656],[0,-0.691,-0.736]]

a_rutting_Fine=[[0,0.136,0.124],[0.367,0,0.31],[0,0.624,0.612],[0,0.436,0.425],[-0.539,0,-0.597],[0,-0.309,-0.321]]
a_rutting_coarse=[[0,0.0958,0.084],[0.327,0,0.269],[0,0.583,0.572],[0,0.396,0.384],[-0.58,0,-0.637],[0,-0.35,-0.361]]
a_rutting_rock=[[0,-0.0199,-0.0317],[0.211,0,0.154],[0,0.468,0.456],[0,0.28,0.269],[-0.695,0,-0.753],[0,-0.465,-0.477]]

a2_fatigue_Fine=[[0,2.43,1.97],[2.51,0,2.14],[0,2.80,2.33],[0,1.97,1.51],[2.15,0,1.78],[0,2.45,1.98]]
a2_fatigue_coarse=[[0,2.06,1.59],[2.14,0,1.77],[0,2.43,1.96],[0,1.60,1.13],[1.77,0,1.40],[0,2.08,1.61]]
a2_fatigue_rock=[[0,2.11,1.64],[2.19,0,1.81],[0,2.47,2.01],[0,1.65,1.18],[1.82,0,1.45],[0,2.12,1.66]]

a_IRI_Fine       = np.array(a_IRI_Fine, dtype=float)
a_IRI_coarse     = np.array(a_IRI_coarse, dtype=float)
a_IRI_rock       = np.array(a_IRI_rock, dtype=float)
a_rutting_Fine   = np.array(a_rutting_Fine, dtype=float)
a_rutting_coarse = np.array(a_rutting_coarse, dtype=float)
a_rutting_rock   = np.array(a_rutting_rock, dtype=float)
a2_fatigue_Fine  = np.array(a2_fatigue_Fine, dtype=float)
a2_fatigue_coarse= np.array(a2_fatigue_coarse, dtype=float)
a2_fatigue_rock  = np.array(a2_fatigue_rock, dtype=float)

A_IRI      = {"Fine": a_IRI_Fine, "Coarse": a_IRI_coarse, "Rock": a_IRI_rock}
A_RUTTING  = {"Fine": a_rutting_Fine, "Coarse": a_rutting_coarse, "Rock": a_rutting_rock}
A2_FATIGUE = {"Fine": a2_fatigue_Fine, "Coarse": a2_fatigue_coarse, "Rock": a2_fatigue_rock}

BASE_OPTIONS = [
    "Asphalt Treated Base",
    "Dense Graded Aggregate Base",
    "Lean Concrete Base",
    "Non Bituminous Treated Base",
    "No Base",
    "Permeable Asphalt Treated Base",
]
BASE_ROW = {label: i for i, label in enumerate(BASE_OPTIONS)}
RFC_COEFFS = {
    "Asphalt Treated Base": (0.331, -0.000228),
    "Dense Graded Aggregate Base": (0.225, -0.000266),
    "Lean Concrete Base": (0.0236, -0.00507),
    "Non Bituminous Treated Base": (0.108, 0.00120),
    "No Base": (1.13, 0.000283),
    "Permeable Asphalt Treated Base": (0.769, 0.000150),
}

STRUCT_OPTIONS = ["Non Overlay Unbound Base", "Non Overlay Bound Base", "Overlay"]
STRUCT_COL     = {label: i for i, label in enumerate(STRUCT_OPTIONS)}
A1_FATIGUE     = {"Non Overlay Unbound Base": -1.06,
                  "Non Overlay Bound Base":   -0.635,
                  "Overlay":                   0.101}

SUBGRADE_OPTIONS = ["Fine", "Coarse", "Rock/Stone"]
FUNCTIONAL_OPTIONS = ["Interstate", "Primary", "Secondary"]

def set_global_fonts(root, family="MS Sans Serif", size=9):
    for name in ("TkDefaultFont","TkTextFont","TkMenuFont","TkFixedFont"):
        try:
            tkfont.nametofont(name).configure(family=family, size=size)
        except tk.TclError:
            pass

def thresholds_for(fc: str):
    if fc == "Interstate":
        return dict(IRI_preservation_threshold=1.5, IRI_rehabilitation_threshold=2.09, IRI_reconstruction_threshold=2.68,
                    rutting_preservation_threshold=5, rutting_rehabilitation_threshold=7.5, rutting_reconstruction_threshold=10,
                    fatigue_preservation_threshold=5, fatigue_rehabilitation_threshold=12.5, fatigue_reconstruction_threshold=20)
    if fc == "Primary":
        return dict(IRI_preservation_threshold=1.5, IRI_rehabilitation_threshold=2.09, IRI_reconstruction_threshold=2.68,
                    rutting_preservation_threshold=5, rutting_rehabilitation_threshold=7.5, rutting_reconstruction_threshold=10,
                    fatigue_preservation_threshold=5, fatigue_rehabilitation_threshold=12.5, fatigue_reconstruction_threshold=20)
    # Secondary
    return dict(IRI_preservation_threshold=1.5, IRI_rehabilitation_threshold=2.09, IRI_reconstruction_threshold=2.68,
                rutting_preservation_threshold=5, rutting_rehabilitation_threshold=7.5, rutting_reconstruction_threshold=10,
                fatigue_preservation_threshold=5, fatigue_rehabilitation_threshold=12.5, fatigue_reconstruction_threshold=20)

# ====== Single window app =====================================================
class PavementEditor:
    def __init__(self, output_root: Path | None = None, create_ui: bool = True):
        # Use caller's path if given, otherwise default to ./outputs
        self.output_root = Path(output_root) if output_root else (Path.cwd() / "outputs")
        self.output_root.mkdir(parents=True, exist_ok=True)

        if create_ui:
            self.root = tk.Tk()
            self.root.title("Strategic Input editor")
            self.root.geometry("900x460+80+60")
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            set_global_fonts(self.root)
        else:
            self.root = None

        # --- compact two-row header --------------------------------------------------
        head = ttk.Frame(self.root, padding=(10,6,10,6))
        head.pack(side="top", fill="x")
        
        # Row 1: section picking + CSV tools
        row1 = ttk.Frame(head)
        row1.pack(side="top", fill="x")
        
        ttk.Label(row1, text="Section").grid(row=0, column=0, sticky="w", padx=(0,8))
        self.v_section = tk.StringVar()
        ttk.Entry(row1, textvariable=self.v_section, width=18).grid(row=0, column=1, sticky="w")
        
        ttk.Label(row1, text="Pick section").grid(row=0, column=2, padx=(12,6))
        self.cbo_section = ttk.Combobox(row1, width=24, state="disabled")
        self.cbo_section.grid(row=0, column=3, sticky="w")
        
        ttk.Button(row1, text="◀ Prev", command=self.section_prev).grid(row=0, column=4, padx=2)
        ttk.Button(row1, text="Next ▶", command=self.section_next).grid(row=0, column=5, padx=2)
        
        row1.columnconfigure(6, weight=1)
        
        ttk.Button(row1, text="Open folder", command=self.open_section_dir).grid(row=0, column=7, padx=6, sticky="e")
        ttk.Button(row1, text="CSV Guide", command=self.show_csv_guide).grid(row=0, column=8, padx=6, sticky="e")
        ttk.Button(row1, text="Load CSV…", command=self.load_csv).grid(row=0, column=9, padx=6, sticky="e")
        
        # Row 2: actions (right-aligned, under row1)
        row2 = ttk.Frame(head)
        row2.pack(side="top", fill="x", pady=(6,0))
        
        for text, cmd in (
            ("Close",  self.root.destroy),
            ("Clear",  self.clear_all),
            ("Save",   self.on_save),
            ("Refresh", self.refresh),
        ):
            ttk.Button(row2, text=text, command=cmd).pack(side="right", padx=6)
        
        # Keep the small hint under the header
        hint = ttk.Label(self.root, text="Tip: Use ‘CSV Guide’ to see the required column titles.", foreground="#555")
        hint.pack(anchor="w", padx=12, pady=(0,6))

        # --- notebook with tabs ----------------------------------------------
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=(0,10))
        self.nb = nb

        self.make_general_tab()
        self.make_structure_tab()
        self.make_environment_tab()
        self.make_layers_tab()
        self.make_costs_tab()
        self.make_user_costs_tab()
        self.make_review_tab()

        self.csv_rows: list[dict] = []
        self.csv_row_idx: int = -1
        self.cbo_section.bind("<<ComboboxSelected>>", lambda e: self._select_from_combo())

        self.state = None

    # ---------- Tab builders --------------------------------------------------
    def make_general_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="General")
        # Analysis period + basic variables
        lfA = ttk.LabelFrame(f, text="Design / Analysis"); lfA.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.v_design_life = tk.IntVar(value=15)
        ttk.Label(lfA, text="Analysis period (years)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(lfA, from_=1, to=200, textvariable=self.v_design_life, width=8).grid(row=0, column=1, sticky="w")

        lfB = ttk.LabelFrame(f, text="Traffic & Geometry"); lfB.grid(row=0, column=1, sticky="nsew")
        self.v_ESAL  = tk.DoubleVar()
        self.v_SN    = tk.DoubleVar()
        self.v_LEN   = tk.DoubleVar()
        self.v_LANES = tk.DoubleVar()
        self.v_AGE_IRI = tk.DoubleVar()
        self.v_INI_IRI = tk.DoubleVar()

        self._add_row(lfB, 0, "ESAL", self.v_ESAL)
        self._add_row(lfB, 1, "Structural Number", self.v_SN)
        self._add_row(lfB, 2, "Pavement Age at Initial IRI", self.v_AGE_IRI)
        self._add_row(lfB, 3, "Initial IRI (m/km)", self.v_INI_IRI)
        self._add_row(lfB, 4, "Section Length (km)", self.v_LEN)
        self._add_row(lfB, 5, "Number of Lanes", self.v_LANES)
        f.columnconfigure((0,1), weight=1)

    def make_structure_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="Structure")

        # Left group: categorical selections
        lf1 = ttk.LabelFrame(f, text="Selections")
        lf1.grid(row=0, column=0, sticky="nsew", padx=(0,10))

        self.v_base  = tk.StringVar(value="Asphalt Treated Base")
        self.v_struct= tk.StringVar(value="Non Overlay Unbound Base")
        self.v_subgr = tk.StringVar(value="Fine")
        self.v_func  = tk.StringVar(value="Interstate")

        self._add_combo(lf1, 0, "Base type", BASE_OPTIONS, self.v_base, self.refresh)
        self._add_combo(lf1, 1, "Pavement structure", STRUCT_OPTIONS, self.v_struct, self.refresh)
        self._add_combo(lf1, 2, "Subgrade type", SUBGRADE_OPTIONS, self.v_subgr, self.refresh)
        self._add_combo(lf1, 3, "Functional class", FUNCTIONAL_OPTIONS, self.v_func, self.refresh)

        # Right group: asphalt thickness + freeze/thaw/precip indices
        lf2 = ttk.LabelFrame(f, text="Materials / Climate")
        lf2.grid(row=0, column=1, sticky="nsew")

        self.v_ATK = tk.DoubleVar()   
        self.v_FT  = tk.DoubleVar()
        self.v_FI  = tk.DoubleVar()
        self.v_CI  = tk.DoubleVar()
        self.v_RAIN= tk.DoubleVar()

        self._add_row(lf2, 0, "Asphalt Thickness (in)", self.v_ATK)
        self._add_row(lf2, 1, "Freeze–Thaw cycles / year", self.v_FT)
        self._add_row(lf2, 2, "Freezing Index (°C·day)", self.v_FI)
        self._add_row(lf2, 3, "Cooling Index (°C·day)",  self.v_CI)
        self._add_row(lf2, 4, "Annual Precipitation (mm)", self.v_RAIN)
        f.columnconfigure((0,1), weight=1)

    def make_environment_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="Coefficients")
        self.v_CO_A   = tk.DoubleVar()
        self.v_CO_B   = tk.DoubleVar()
        self.v_CO_SUB = tk.DoubleVar()

        lf = ttk.LabelFrame(f, text="Layer coefficients (AASHTO)")
        lf.pack(fill="x")
        self._add_row(lf, 0, "Asphalt layer coeff.", self.v_CO_A)
        self._add_row(lf, 1, "Base layer coeff.",    self.v_CO_B)
        self._add_row(lf, 2, "Subbase layer coeff.", self.v_CO_SUB)

    def make_layers_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="Layers / Overlay")

        lf1 = ttk.LabelFrame(f, text="Layers")
        lf1.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.v_TH_B   = tk.DoubleVar()
        self.v_TH_SUB = tk.DoubleVar()
        self.v_DCO_B  = tk.DoubleVar()
        self.v_DCO_SUB= tk.DoubleVar()
        self._add_row(lf1, 0, "Base thickness (in)", self.v_TH_B)
        self._add_row(lf1, 1, "Subbase thickness (in)", self.v_TH_SUB)
        self._add_row(lf1, 2, "Drainage coeff. base", self.v_DCO_B)
        self._add_row(lf1, 3, "Drainage coeff. subbase", self.v_DCO_SUB)

        lf2 = ttk.LabelFrame(f, text="Overlay")
        lf2.grid(row=0, column=1, sticky="nsew")
        self.v_TH_OVER = tk.DoubleVar(value=0)
        self.v_CO_ASTL = tk.DoubleVar(value=0)
        self._add_row(lf2, 0, "Overlay thickness (in)", self.v_TH_OVER)
        self._add_row(lf2, 1, "Asphalt structural coeff.", self.v_CO_ASTL)
        f.columnconfigure((0,1), weight=1)

    def make_costs_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="Agency Costs")

        self.v_Construction_Cost  = tk.DoubleVar()
        self.v_Maintenance_Cost   = tk.DoubleVar()
        self.v_Rehabilitation_Cost= tk.DoubleVar()
        self.v_Reconstruction_Cost= tk.DoubleVar()
        self.v_Interest_rate      = tk.DoubleVar()
        self.v_Inflation_rate     = tk.DoubleVar()

        self._add_row(f, 0, "Initial Construction ($/lane-km)", self.v_Construction_Cost)
        self._add_row(f, 1, "Preventive Maintenance ($/lane-km)", self.v_Maintenance_Cost)
        self._add_row(f, 2, "Rehabilitation ($/lane-km)", self.v_Rehabilitation_Cost)
        self._add_row(f, 3, "Reconstruction ($/lane-km)", self.v_Reconstruction_Cost)
        self._add_row(f, 4, "Interest rate (e.g., 0.05)", self.v_Interest_rate)
        self._add_row(f, 5, "Inflation rate (e.g., 0.03)", self.v_Inflation_rate)
        f.columnconfigure(1, weight=1)

    def make_user_costs_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="User Costs")

        # left col
        lfL = ttk.LabelFrame(f, text="Traffic & speed")
        lfL.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        self.v_UC_AADT  = tk.StringVar()
        self.v_UC_TRUCK = tk.StringVar()
        self.v_FFS      = tk.DoubleVar()
        self.v_MINSPD   = tk.DoubleVar()
        self.v_IRI0     = tk.StringVar()
        self.v_K        = tk.DoubleVar()

        self._add_row(lfL, 0, "AADT (veh/day)", self.v_UC_AADT)
        self._add_row(lfL, 1, "% Trucks (12 or 0.12)", self.v_UC_TRUCK)
        self._add_row(lfL, 2, "Free-flow speed (km/h)", self.v_FFS)
        self._add_row(lfL, 3, "Minimum speed (km/h)", self.v_MINSPD)
        self._add_row(lfL, 4, "IRI₀ for TTC", self.v_IRI0)
        self._add_row(lfL, 5, "k (km/h per 1 IRI)", self.v_K)

        # right col
        lfR = ttk.LabelFrame(f, text="Value of time & VOC")
        lfR.grid(row=0, column=1, sticky="nsew")
        self.v_VOC_CAR   = tk.DoubleVar()
        self.v_VOC_TRUCK = tk.DoubleVar()
        self.v_OCC_CAR   = tk.DoubleVar()
        self.v_OCC_TRUCK = tk.DoubleVar()
        self.v_VOT_CAR   = tk.DoubleVar()
        self.v_VOT_TRUCK = tk.DoubleVar()

        self._add_row(lfR, 0, "VOC — car ($/veh-km)", self.v_VOC_CAR)
        self._add_row(lfR, 1, "VOC — truck ($/veh-km)", self.v_VOC_TRUCK)
        self._add_row(lfR, 2, "Occupancy — car (pax/veh)", self.v_OCC_CAR)
        self._add_row(lfR, 3, "Occupancy — truck (pax/veh)", self.v_OCC_TRUCK)
        self._add_row(lfR, 4, "VOT — car ($/h)", self.v_VOT_CAR)
        self._add_row(lfR, 5, "VOT — truck ($/h)", self.v_VOT_TRUCK)
        f.columnconfigure((0,1), weight=1)

    def make_review_tab(self):
        f = ttk.Frame(self.nb, padding=12); self.nb.add(f, text="Review")
        self.review_txt = tk.Text(f, height=20)
        self.review_txt.pack(fill="both", expand=True)
        ttk.Button(f, text="Refresh summary", command=self.refresh).pack(anchor="e", pady=8)

    # ---------- helpers -------------------------------------------------------
    def _add_row(self, parent, r, label, var):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(8,6), pady=3)
        if isinstance(var, (tk.IntVar, tk.DoubleVar, tk.StringVar)):
            e = ttk.Entry(parent, textvariable=var, width=18)
        else:
            e = ttk.Entry(parent, width=18)
        e.grid(row=r, column=1, sticky="w", pady=3)

    def _add_combo(self, parent, r, label, options, var, callback=None):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(8,6), pady=3)
        cb = ttk.Combobox(parent, values=options, textvariable=var, width=34, state="readonly")
        cb.grid(row=r, column=1, sticky="w", pady=3)
        if callback:
            cb.bind("<<ComboboxSelected>>", lambda e: callback())

    # ---------- logic: compute dependent coefficients -------------------------
    def compute_coeffs(self):
        base = self.v_base.get()
        struct = self.v_struct.get()
        subgr = self.v_subgr.get()

        # row by base
        row = BASE_ROW[base]
        IRI_FI, IRI_CO, IRI_RO = a_IRI_Fine[row], a_IRI_coarse[row], a_IRI_rock[row]
        Rut_FI, Rut_CO, Rut_RO = a_rutting_Fine[row], a_rutting_coarse[row], a_rutting_rock[row]
        Fat_FI, Fat_CO, Fat_RO = a2_fatigue_Fine[row], a2_fatigue_coarse[row], a2_fatigue_rock[row]
        beta, gama = RFC_COEFFS[base]

        # column by structure
        col = STRUCT_COL[struct]
        a_IRI_FI = IRI_FI[col]; a_IRI_CO = IRI_CO[col]; a_IRI_RO = IRI_RO[col]
        a_rut_FI = Rut_FI[col]; a_rut_CO = Rut_CO[col]; a_rut_RO = Rut_RO[col]
        a2_fat_FI= Fat_FI[col]; a2_fat_CO= Fat_CO[col]; a2_fat_RO= Fat_RO[col]
        a1_fatigue = A1_FATIGUE[struct]

        # bundle by subgrade choice
        if subgr == "Fine":
            a_IRI = a_IRI_FI; a_rutting = a_rut_FI; a2_fatigue = a2_fat_FI
        elif subgr == "Coarse":
            a_IRI = a_IRI_CO; a_rutting = a_rut_CO; a2_fatigue = a2_fat_CO
        else:
            a_IRI = a_IRI_RO; a_rutting = a_rut_RO; a2_fatigue = a2_fat_RO

        return dict(
            IRI_FI=IRI_FI, IRI_CO=IRI_CO, IRI_RO=IRI_RO,
            Rutting_FI=Rut_FI, Rutting_CO=Rut_CO, Rutting_RO=Rut_RO,
            Fatigue_FI=Fat_FI, Fatigue_CO=Fat_CO, Fatigue_RO=Fat_RO,
            a_IRI_FI=a_IRI_FI, a_IRI_CO=a_IRI_CO, a_IRI_RO=a_IRI_RO,
            a_rutting_FI=a_rut_FI, a_rutting_CO=a_rut_CO, a_rutting_RO=a_rut_RO,
            a2_fatigue_FI=a2_fat_FI, a2_fatigue_CO=a2_fat_CO, a2_fatigue_RO=a2_fat_RO,
            a_IRI=a_IRI, a_rutting=a_rutting, a2_fatigue=a2_fatigue,
            beta=beta, gama=gama, a1_fatigue=a1_fatigue
        )

    # ---------- actions --------------------------------------------------------
    def refresh(self):
        # Compose a quick summary so the user can see everything at once
        coeffs = self.compute_coeffs()
        fc = self.v_func.get()
        thr = thresholds_for(fc)
        txt = []
        txt.append(f"Section: {self.v_section.get()}")
        txt.append(f"Base: {self.v_base.get()} | Structure: {self.v_struct.get()} | Subgrade: {self.v_subgr.get()} | Class: {fc}")
        txt.append(f"Analysis period: {self.v_design_life.get()} years")
        txt.append(f"ESAL={self.v_ESAL.get():.0f}, SN={self.v_SN.get():.2f}, Age@IRI={self.v_AGE_IRI.get():.2f} yr, IRI0={self.v_INI_IRI.get():.3f}")
        txt.append(f"Length={self.v_LEN.get():.3f} km, Lanes={self.v_LANES.get():.0f}")
        txt.append("Coefficients (selected): " +
                   f"a_IRI={coeffs['a_IRI']:.3f}, a_rutting={coeffs['a_rutting']:.3f}, a2_fatigue={coeffs['a2_fatigue']:.3f}, a1_fatigue={coeffs['a1_fatigue']:.3f}")
        txt.append("Thresholds: " + ", ".join([f"{k}={v}" for k,v in thr.items()]))
        if hasattr(self, "review_txt"):
            self.review_txt.delete("1.0","end")
            self.review_txt.insert("1.0", "\n".join(txt))

    def clear_all(self):
        if messagebox.askyesno("Reset", "Reset all fields to defaults?"):
            self.__init__()  # cheap reset
            self.root.update()

    def on_save(self, silent: bool = False):
        coeffs = self.compute_coeffs()
        fc = self.v_func.get()
        thr = thresholds_for(fc)
        n = int(self.v_design_life.get())
    
        state = {}
        state["Section_Name"] = self.v_section.get()
        state["S1"] = self.v_base.get()
        state["S2"] = self.v_struct.get()
        state["S3"] = self.v_subgr.get()
        state["S4"] = fc
        state["Design_life"] = n
        state["AGE"] = np.arange(n + 1, dtype=int)
    
        state["answer"] = [
            float(self.v_ESAL.get()), float(self.v_SN.get()),
            float(self.v_ATK.get()), float(self.v_FT.get()),
            float(self.v_FI.get()), float(self.v_CI.get()),
            float(self.v_RAIN.get()), float(self.v_AGE_IRI.get()),
            float(self.v_INI_IRI.get()), float(self.v_LEN.get()),
            float(self.v_LANES.get()),
        ]
    
        state["Coefficients_changes"] = [self.v_CO_A.get(), self.v_CO_B.get(), self.v_CO_SUB.get()]
        state["Layer_data"]          = [self.v_TH_B.get(), self.v_TH_SUB.get(), self.v_DCO_B.get(), self.v_DCO_SUB.get()]
        state["overlay_input"]       = [self.v_TH_OVER.get(), self.v_CO_ASTL.get()]
        state["agency_cost_data"]    = [self.v_Construction_Cost.get(), self.v_Maintenance_Cost.get(),
                                        self.v_Rehabilitation_Cost.get(), self.v_Reconstruction_Cost.get(),
                                        self.v_Interest_rate.get(), self.v_Inflation_rate.get()]
    
        uc_aadt = self.v_UC_AADT.get().strip()
        uc_aadt_val = None if uc_aadt=="" else float(uc_aadt)

        # make/create the section directory
        sec_name = safe_folder_name(self.v_section.get())
        out_dir = (self.output_root / sec_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.current_section_dir = out_dir  # remember it for "Open folder"
        
        # >>> export for the rest of your script <<<
        g = globals()
        g["out_dir"] = out_dir
        g["outdir"] = out_dir
        # (optional but convenient for your later lane-km scaling)
        globals()["LEN_KM"] = float(self.v_LEN.get())
        globals()["LANE_COUNT"] = int(round(self.v_LANES.get()))
        
        def pct_or_decimal(s):
            if s.strip() == "":
                return 0.0
            v = float(s)
            return v / 100.0 if v > 1.0 else v

        state["user_cost_data"] = dict(
            UC_AADT          = uc_aadt_val,
            UC_TRUCK_SHARE   = pct_or_decimal(self.v_UC_TRUCK.get()),
            UC_FFS_KPH       = self.v_FFS.get(),
            UC_MIN_SPEED     = self.v_MINSPD.get(),
            UC_IRI0          = (float(self.v_INI_IRI.get()) if self.v_IRI0.get().strip()=="" else float(self.v_IRI0.get())),
            UC_K_KPH_PER_IRI = self.v_K.get(),
            UC_VOC_CAR       = self.v_VOC_CAR.get(),
            UC_VOC_TRK       = self.v_VOC_TRUCK.get(),
            UC_OCC_CAR       = self.v_OCC_CAR.get(),
            UC_OCC_TRK       = self.v_OCC_TRUCK.get(),
            UC_VOT_CAR       = self.v_VOT_CAR.get(),
            UC_VOT_TRK       = self.v_VOT_TRUCK.get(),
        )
                    
        # store state
        self.state = state
    
        # export to globals
        g = globals()
        g.update(coeffs)      
        g.update(thr)         
        g.update({
            "Section_Name": state["Section_Name"],
            "S1": state["S1"], "S2": state["S2"], "S3": state["S3"], "S4": state["S4"],
            "Design_life": state["Design_life"], "AGE": state["AGE"],
            "answer": state["answer"], "Coefficients_changes": state["Coefficients_changes"],
            "Layer_data": state["Layer_data"], "overlay_input": state["overlay_input"],
            "agency_cost_data": state["agency_cost_data"], "user_cost_data": state["user_cost_data"],
        })
    
        # from 'answer' list -> simple scalars
        ESAL, SN, ATK, FT, FI, CI, RAIN, AGE_IRI, INI_IRI, LEN, LANES = state["answer"]
        g.update({
            "ESAL": float(ESAL), "SN": float(SN),
            "AT": float(ATK),    # map ATK -> AT for your formulas
            "FT": float(FT), "FI": float(FI), "CI": float(CI),
            "RAIN": float(RAIN), "AGE_IRI": float(AGE_IRI), "INI_IRI": float(INI_IRI),
            "LEN": float(LEN), "LANES": float(LANES),
        })
    
        # Coefficients_changes
        CO_A, CO_B, CO_SUB = state["Coefficients_changes"]
        g.update({"CO_A": float(CO_A), "CO_B": float(CO_B), "CO_SUB": float(CO_SUB)})
    
        # Layer_data
        TH_B, TH_SUB, DCO_B, DCO_SUB = state["Layer_data"]
        g.update({"TH_B": float(TH_B), "TH_SUB": float(TH_SUB), "DCO_B": float(DCO_B), "DCO_SUB": float(DCO_SUB)})
    
        # overlay_input
        TH_OVER, CO_ASTL = state["overlay_input"]
        g.update({"TH_OVER": float(TH_OVER), "CO_ASTL": float(CO_ASTL)})
    
        # agency_cost_data
        (Construction_Cost, Maintenance_Cost, Rehabilitation_Cost,
         Reconstruction_Cost, Interest_rate, Inflation_rate) = state["agency_cost_data"]
        g.update({
            "Construction_Cost": float(Construction_Cost),
            "Maintenance_Cost": float(Maintenance_Cost),
            "Rehabilitation_Cost": float(Rehabilitation_Cost),
            "Reconstruction_Cost": float(Reconstruction_Cost),
            "Interest_rate": float(Interest_rate),
            "Inflation_rate": float(Inflation_rate),
        })
    
        # user_cost_data (dict): UC_* etc.
        for k, v in state["user_cost_data"].items():
            g[k] = v
        
        if not silent:
            messagebox.showinfo("Saved", "Inputs captured and exported to globals.")

    def load_csv(self):
        p = fd.askopenfilename(title="Open inputs CSV",
                               filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if not p:
            return
    
        rows = None
        # Fast path: pandas (robust on big CSVs with blank tails)
        try:
            df = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[],
                             skip_blank_lines=True)
            # Drop completely empty columns (Excel cruft)
            df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
            # Drop completely empty rows
            df = df[df.apply(lambda s: any(str(x).strip() for x in s), axis=1)]
            rows = df.to_dict(orient="records")
        except Exception:
            # Fallback: Python csv with forced comma delimiter
            import csv
            with open(p, encoding="utf-8-sig", errors="replace", newline="") as f:
                try:
                    sample = f.read(65536); f.seek(0)
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
                    delim = dialect.delimiter
                except csv.Error:
                    delim = ","  # force comma if sniffer can’t decide
                rdr = csv.DictReader(f, delimiter=delim, skipinitialspace=True)
                rows = []
                for r in rdr:
                    if r and any((str(v).strip() for v in r.values())):  # skip empty rows
                        rows.append(r)
    
        # Normalize headers → internal keys you already use
        rows = [self._translate_headers(r) for r in rows]
    
        self.csv_rows = rows
        names = []
        for i, r in enumerate(rows, start=1):
            nm = r.get("Section_Name") or r.get("Section name") or f"Row {i}"
            names.append(f"{i:02d} — {nm}")
    
        self.cbo_section.config(state="readonly", values=names)
        self.cbo_section.current(0)
        self.csv_row_idx = 0
        updated = self._fill_from_dict(rows[0])
        self.refresh()
        self.on_save(silent=True)
    
        msg = (f"Found {len(rows)} sections. Loaded #1: {names[0]}\n"
               f"Use the drop-down / Prev / Next to switch sections.") if len(rows) > 1 \
              else f"Filled {updated} fields from:\n{p}"
        tk.messagebox.showinfo("CSV loaded", msg)

    def show_csv_guide(self):
        win = tk.Toplevel(self.root)
        win.title("CSV Guide")
        win.geometry("520x420+120+90")
        ttk.Label(win, text="Upload a one-row CSV with these column titles:",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=12, pady=(12,6))
    
        # list view (easier to read)
        frm = ttk.Frame(win); frm.pack(fill="both", expand=True, padx=12, pady=(0,12))
        text = tk.Text(frm, height=14, wrap="word")
        sb = ttk.Scrollbar(frm, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=sb.set)
        text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
    
        text.insert("1.0", "\n".join(CSV_TITLES_FRIENDLY))
        text.configure(state="disabled")
    
        # header line (comma-separated) + copy button
        hdr_line = ",".join(CSV_TITLES_FRIENDLY)
    
        btns = ttk.Frame(win); btns.pack(fill="x", padx=12, pady=(0,12))
        def copy_headers():
            self.root.clipboard_clear()
            self.root.clipboard_append(hdr_line)
            messagebox.showinfo("Copied", "Header line copied to clipboard.")
    
        ttk.Button(btns, text="Copy header line", command=copy_headers).pack(side="left")
        ttk.Button(btns, text="Save CSV Template…", command=self.save_csv_template).pack(side="left", padx=8)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right")

    def save_csv_template(self):
        p = fd.asksaveasfilename(
            title="Save CSV template",
            defaultextension=".csv",
            initialfile="pavement_input_template.csv",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")]
        )
        if not p:
            return
        import csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CSV_TITLES_FRIENDLY)  # header only
        messagebox.showinfo("Saved", f"Template saved:\n{p}")

    def _translate_headers(self, raw: dict) -> dict:
        """Map friendly CSV titles to internal keys (TITLES_TO_KEYS), keep exact keys if already internal."""
        out = {}
        for k, v in raw.items():
            if not k: 
                continue
            nk = _norm_key(k)
            std = TITLES_TO_KEYS.get(nk)
            out[std or k] = (v.strip() if isinstance(v, str) else v)
        return out
    
    def _fill_from_dict(self, data: dict) -> int:
        """Set Tk variables from a normalized dict (internal keys). Returns #updated fields."""
        M = {
            "Section_Name": self.v_section,
            "S1": (self.v_base, ("Asphalt Treated Base","Dense Graded Aggregate Base",
                                 "Lean Concrete Base","Non Bituminous Treated Base","No Base","Permeable Asphalt Treated Base")),
            "S2": (self.v_struct, ("Non Overlay Unbound Base","Non Overlay Bound Base","Overlay")),
            "S3": (self.v_subgr,  ("Fine","Coarse","Rock/Stone")),
            "S4": (self.v_func,   ("Interstate","Primary","Secondary")),
            "Design_life": self.v_design_life,
            "ESAL": self.v_ESAL, "SN": self.v_SN, "ATK": self.v_ATK, "FT": self.v_FT,
            "FI": self.v_FI, "CI": self.v_CI, "RAIN": self.v_RAIN,
            "AGE_IRI": self.v_AGE_IRI, "INI_IRI": self.v_INI_IRI, "LEN": self.v_LEN, "LANES": self.v_LANES,
            "CO_A": self.v_CO_A, "CO_B": self.v_CO_B, "CO_SUB": self.v_CO_SUB,
            "TH_B": self.v_TH_B, "TH_SUB": self.v_TH_SUB, "DCO_B": self.v_DCO_B, "DCO_SUB": self.v_DCO_SUB,
            "TH_OVER": self.v_TH_OVER, "CO_ASTL": self.v_CO_ASTL,
            "Construction_Cost": self.v_Construction_Cost, "Maintenance_Cost": self.v_Maintenance_Cost,
            "Rehabilitation_Cost": self.v_Rehabilitation_Cost, "Reconstruction_Cost": self.v_Reconstruction_Cost,
            "Interest_rate": self.v_Interest_rate, "Inflation_rate": self.v_Inflation_rate,
            "UC_AADT": self.v_UC_AADT,"UC_TRUCK_SHARE": self.v_UC_TRUCK,"UC_FFS_KPH": self.v_FFS,"UC_MIN_SPEED": self.v_MINSPD,
            "UC_IRI0": self.v_IRI0, "UC_K_KPH_PER_IRI": self.v_K,"UC_VOC_CAR": self.v_VOC_CAR,"UC_VOC_TRK": self.v_VOC_TRUCK,
            "UC_OCC_CAR": self.v_OCC_CAR,"UC_OCC_TRK": self.v_OCC_TRUCK,"UC_VOT_CAR": self.v_VOT_CAR,"UC_VOT_TRK": self.v_VOT_TRUCK,
        }
        
        def to_float(x):
            try: return float(x)
            except: return None
    
        updated = 0
        for k, v in data.items():
            if k not in M or v == "":
                continue
            target = M[k]
            if isinstance(target, tuple):
                var, options = target
                # exact or case-insensitive match
                val = str(v)
                if val in options:
                    var.set(val); updated += 1
                else:
                    for opt in options:
                        if val.lower() == opt.lower():
                            var.set(opt); updated += 1; break
            else:
                if isinstance(target, tk.StringVar):
                    target.set(str(v)); updated += 1
                else:
                    f = to_float(v)
                    if f is not None:
                        target.set(f); updated += 1
    
        return updated

    def _apply_row_index(self, idx: int):
        if not self.csv_rows: 
            return
        idx = max(0, min(idx, len(self.csv_rows) - 1))
        self.csv_row_idx = idx
        self.cbo_section.current(idx)
        updated = self._fill_from_dict(self.csv_rows[idx])
        self.refresh()
        self.on_save(silent=True)
    
    def _select_from_combo(self):
        self._apply_row_index(self.cbo_section.current())
    
    def section_prev(self):
        if self.csv_rows:
            self._apply_row_index(self.csv_row_idx - 1)
    
    def section_next(self):
        if self.csv_rows:
            self._apply_row_index(self.csv_row_idx + 1)

    def open_section_dir(self):
        sec_name = safe_folder_name(self.v_section.get())
        d = (self.output_root / sec_name)
        d.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(d))
            elif sys.platform == "darwin":
                import subprocess; subprocess.run(["open", str(d)], check=False)
            else:
                import subprocess; subprocess.run(["xdg-open", str(d)], check=False)
        except Exception:
            messagebox.showinfo("Folder", f"Folder is at:\n{d}")

    def load_csv_rows_from_path(self, p: str) -> list[dict]:
        """Load and normalize rows from a CSV path (header or key,value). No dialogs, no popups."""
        rows = []
        with open(p, newline="", encoding="utf-8-sig") as f:
            sample = f.read(1024); f.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                rdr = csv.DictReader(f)
                raw = [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
                       for row in rdr]
                rows = [self._translate_headers(r) for r in raw]
            else:
                rdr = csv.reader(f)
                tmp = {}
                for row in rdr:
                    if len(row) >= 2:
                        tmp[row[0].strip()] = row[1].strip()
                rows = [self._translate_headers(tmp)]
        return rows

    # ---------- run -----------------------------------------------------------
    def run(self):
        self.refresh()
        self.root.mainloop()
        return self.state

def run_wizard_one_window():
    app = PavementEditor()
    state = app.run()
    return state

state = run_wizard_one_window()


# In[6]:


IRI = []
for k in range(len(AGE)):
    DELTA = round(
        a_IRI
        + 0.115 * (math.log10(ESAL)/SN)
        + 7.9e-3 * AGE_IRI
        - 4.33e-5 * CI
        + 2.28e-6 * FI
        + 5.9e-5 * RAIN
        + 2.21e-4 * FT
        + 8.59e-5 * AT
        - 5.39e-3 * (math.log10(ESAL)/SN) * AGE_IRI
        + 1.77e-6 * AGE_IRI * CI
        + 4.55e-6 * AGE_IRI * FI
        + 0.643 * INI_IRI
        - 2.4e-6 * AGE_IRI * RAIN
        - 1.09e-5 * AGE_IRI * FT
        - math.log(INI_IRI + 0.1),
        4
    )

    iri_k = round(
        math.exp(
            a_IRI - DELTA
            + 0.115 * (math.log10(ESAL)/SN)
            + 3.29e-2 * AGE[k]
            - 4.33e-5 * CI
            + 2.28e-6 * FI
            + 5.9e-5 * RAIN
            + 2.21e-4 * FT
            + 8.59e-5 * AT
            - 5.39e-3 * (math.log10(ESAL)/SN) * AGE[k]
            + 1.77e-6 * AGE[k] * CI
            + 4.55e-6 * AGE[k] * FI
            + 0.643 * INI_IRI
            - 2.5e-2 * AGE_IRI
            - 2.4e-6 * AGE[k] * RAIN
            - 1.09e-5 * AGE[k] * FT
        ) - 0.1,
        4
    )
    IRI.append(iri_k)

rutting = []
for k in range(len(AGE)):
    r_k = round(
        math.exp(
            a_rutting
            + 0.503 * math.log(AGE[k] + 0.1)
            + 3.37e-4 * CI
            + 1.22e-5 * RAIN
            + 3.48e-3 * FT
            + 2.98e-7 * FI * RAIN
            - 8.44e-2 * (math.log10(ESAL)/SN) * (math.log(AGE[k] + 0.1))
            - 1.42e-4 * (math.log(AGE[k] + 0.1)) * CI
            - 1.38e-5 * (math.log(AGE[k] + 0.1)) * FI
            + float(beta) * (math.log10(ESAL)/SN)
            + float(gama) * FI
        ) - 0.1,
        4
    )
    rutting.append(r_k)

fatigue = []
p = [
    0.513 - 6.1e-2 * (math.log10(ESAL)/SN) - 1.15e-4 * CI - 1.44e-4 * FI,
    -math.log(0.7/(1-0.7)) + a1_fatigue + 0.917 * (math.log10(ESAL)/SN) - 1.12e-2 * FT
    + 1.06e-4 * FI - 2.21e-4 * CI - 1.01e-3 * RAIN
]
roots = np.roots(p)
first_crack_age = float(roots[0]) if roots.size > 0 else 0.0
if first_crack_age < 0:
    first_crack_age = 0.0

for k in range(len(AGE)):
    if AGE[k] < first_crack_age:
        f_k = 0
    else:
        fca = first_crack_age
        globals()["fca"] = first_crack_age
        f_k = np.round(
            math.exp(
                a2_fatigue
                + 7.63e-2 * (math.log10(ESAL)/SN)
                + 0.737 * math.log(AGE[k] - first_crack_age + 0.1)
                - 1.04e-3 * CI
                - 4.12e-4 * FI
                + 2.03e-5 * RAIN
                - 1.12e-2 * FT
                + 2.07e-4 * RAIN * math.log(AGE[k] - first_crack_age + 0.1)
            ) - 0.1,
            4
        )
    fatigue.append(f_k)

def _prefix_len(series, threshold, design_life):
    """
    Count consecutive years from year 1 where value <= threshold.
    Returns the last compliant year index (relative to year 1).
    """
    hard_limit = min(int(design_life), len(series) - 1)
    x = 1
    while x <= hard_limit and series[x] <= threshold:
        x += 1
    return x - 1

def _fc(x, design_life):
    return min(int(x) + 1, int(design_life))
    
def rsi_three_stage(series, t_pres, t_rehab, t_recon, design_life):
    """
    Returns scalar RSIs (pres, rehab, recon) using FIRST-CROSSING policy.
    """
    if len(series) <= 1:
        raise ValueError("series must have length >= 2 (uses series[1]).")

    v1 = series[1]

    if v1 <= t_pres:
        pres  = _fc(_prefix_len(series, t_pres,  design_life), design_life)
        rehab = _fc(_prefix_len(series, t_rehab, design_life), design_life)
        recon = _fc(_prefix_len(series, t_recon, design_life), design_life)

    elif (t_pres < v1) and (v1 < t_rehab):
        pres  = 1
        rehab = _fc(_prefix_len(series, t_rehab, design_life), design_life)
        recon = _fc(_prefix_len(series, t_recon, design_life), design_life)

    elif (t_rehab <= v1) and (v1 < t_recon):
        pres  = 1
        rehab = 1
        recon = _fc(_prefix_len(series, t_recon, design_life), design_life)  # <-- +1 added

    else:
        pres = rehab = recon = 1

    return pres, rehab, recon

# Distress-specific RSIs
RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction = rsi_three_stage(
    IRI, IRI_preservation_threshold, IRI_rehabilitation_threshold, IRI_reconstruction_threshold, Design_life
)
RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction = rsi_three_stage(
    rutting, rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold, Design_life
)
RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction = rsi_three_stage(
    fatigue, fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold, Design_life
)


# In[3]:


# ---- Math helpers ----
def compute_delta(a_IRI, ESAL, SN, AGE_val, CI, FI, RAIN, FT, AT, INI_IRI):
    k = math.log10(ESAL) / SN
    return (
        a_IRI
        + 0.115 * k
        + 7.9e-3 * AGE_val
        - 4.33e-5 * CI
        + 2.28e-6 * FI
        + 5.9e-5 * RAIN
        + 2.21e-4 * FT
        + 8.59e-5 * AT
        - 5.39e-3 * k * AGE_val
        + 1.77e-6 * AGE_val * CI
        + 4.55e-6 * AGE_val * FI
        + 0.643 * INI_IRI
        - 2.4e-6 * AGE_val * RAIN
        - 1.09e-5 * AGE_val * FT
        - math.log(INI_IRI + 0.1)
    )

def iri_series(a_IRI, ESAL, SN, CI, FI, RAIN, FT, AT, miri_age, miri, AGE, INI_IRI):
    """Produces exactly the same IRI values you compute now (per t in AGE)."""
    k = math.log10(ESAL) / SN
    out = []
    for t in AGE:
        DELTA = compute_delta(a_IRI, ESAL, SN, miri_age, CI, FI, RAIN, FT, AT, miri)
        val = math.exp(
            a_IRI - DELTA
            + 0.115 * k
            + 3.29e-2 * t
            - 4.33e-5 * CI
            + 2.28e-6 * FI
            + 5.9e-5 * RAIN
            + 2.21e-4 * FT
            + 8.59e-5 * AT
            - 5.39e-3 * k * t
            + 1.77e-6 * t * CI
            + 4.55e-6 * t * FI
            + 0.643 * miri
            - 2.5e-2 * miri_age
            - 2.4e-6 * t * RAIN
            - 1.09e-5 * t * FT
        ) - 0.1
        out.append(round(val, 4))
    return out

def rutting_series(a_rutting, ESAL, SN, CI, FI, RAIN, FT, AGE, beta, gamma):
    k = math.log10(ESAL) / SN
    out = []
    for t in AGE:
        lt = math.log(t + 0.1)
        val = math.exp(
            a_rutting
            + 0.503 * lt
            + 3.37e-4 * CI
            + 1.22e-5 * RAIN
            + 3.48e-3 * FT
            + 2.98e-7 * FI * RAIN
            - 8.44e-2 * k * lt
            - 1.42e-4 * lt * CI
            - 1.38e-5 * lt * FI
            + beta * k
            + gamma * FI
        ) - 0.1
        out.append(round(val, 4))
    return out

def fatigue_series(a2_fatigue, ESAL, SN, CI, FI, RAIN, FT, AGE, first_crack_age):
    k = math.log10(ESAL) / SN
    out = []
    for t in AGE:
        if t < first_crack_age:
            out.append(0)
        else:
            lt = math.log(t - first_crack_age + 0.1)
            val = math.exp(
                a2_fatigue
                + 7.63e-2 * k
                + 0.737 * lt
                - 1.04e-3 * CI
                - 4.12e-4 * FI
                + 2.03e-5 * RAIN
                - 1.12e-2 * FT
                + 2.07e-4 * RAIN * lt
            ) - 0.1
            out.append(round(val, 4))
    return out

def rsi_lists_from_series(series, thresholds, start, design_life):
    """
    Build (pres, rehab, recon) RSI lists for one distress series and its 3 thresholds.
    Stops safely at the end of the preserved window to avoid IndexError.
    """
    start = int(start)
    design_life = int(design_life)
    res = [[] for _ in thresholds]

    # length of the subsequence you passed in (e.g., Design_life-start+1)
    n = len(series)
    if n == 0 or design_life <= start:
        return res

    s0 = series[0]

    # helper: walk forward while under thr, respecting bounds
    def _walk(i, thr):
        x = 0
        limit = n  # safe upper bound on x (== len(series))
        while x < limit and series[x] <= thr:
            res[i].append(x + start + 1)
            x += 1

    for i, thr in enumerate(thresholds):
        if s0 <= thr:
            _walk(i, thr)
        elif (i == 0 and s0 < thresholds[1]) or (i == 1 and s0 < thresholds[2]):
            res[i].append(1 + start)
            _walk(i, thr)
        else:
            res[i].append(1 + start)

    return res
    
# --- Shared helpers (module-level) ---
def _find_y(arr, baseline, design_life=None):
    """Return the first index y (scanning backward) such that arr[y] >= baseline.
    """
    n = len(arr)
    x = int(design_life if design_life is not None else globals().get("Design_life", n - 1))
    x = max(0, min(x, n - 1))
    while x >= 0 and arr[x] >= baseline:
        if arr[x] == baseline:
            return x
        x -= 1
    y = x + 1
    if y < 0: y = 0
    if y >= n: y = n - 1
    return y

def _windowed_area_per_year(ir, fa, ru, prev_year, curr_year):
    """
    Deterministic per-year area on (prev_year, curr_year]:
    For a segment of length L = curr_year - prev_year, use the first L+1 points of each
    (already-aligned) series and compute trapezoids per 1-year step.
    """
    prev_year = int(prev_year); curr_year = int(curr_year)
    L = curr_year - prev_year
    if L <= 0:
        return []

    ir = np.asarray(ir, float)[:L+1] / float(IRI_reconstruction_threshold)
    fa = np.asarray(fa, float)[:L+1] / float(fatigue_reconstruction_threshold)
    ru = np.asarray(ru, float)[:L+1] / float(rutting_reconstruction_threshold)

    # per-year trapezoid for each series, then sum
    per_year = 0.5 * ((ir[:-1] + ir[1:]) + (fa[:-1] + fa[1:]) + (ru[:-1] + ru[1:]))
    return [np.round(x, 2) for x in per_year.tolist()]

# User cost helper
NPV_user = 0.0
user_cost_by_year = {}  # year -> annual user cost ($/yr)
# Global multiplier for sensitivity analysis (1.0 = base case)
UC_SCALE = 1

def _uc_speed_from_iri(iri, ffs, iri0, k, vmin):
    penalty = max(0.0, iri - iri0) * k
    return max(vmin, ffs - penalty)
    
def _uc_daily_voc_ttc(iri):
    """
    Daily user costs for the whole section at a given IRI: VOC, TTC
    """
    # If AADT is missing or zero, no user cost
    if UC_AADT is None or float(UC_AADT) <= 0:
        return 0.0, 0.0

    iri = float(iri)

    # ---- 1. Daily VMT for the section --------------------------------------
    vmt_total = float(UC_AADT) * float(LEN)
    vmt_trk   = vmt_total * float(UC_TRUCK_SHARE)
    vmt_car   = vmt_total - vmt_trk

    # ---- 2. IRI-dependent VOC per veh-km -----------------------------------
    iri_ref   = float(UC_IRI0)     # reference smoothness (m/km)
    gamma_car = 0.10    # ~10% more VOC per 1 m/km above iri_ref
    gamma_trk = 0.05    # trucks a bit less sensitive

    over_ref      = max(0.0, iri - iri_ref)
    factor_car    = 1.0 + gamma_car * over_ref
    factor_trk    = 1.0 + gamma_trk * over_ref

    voc_car_km = float(UC_VOC_CAR) * factor_car
    voc_trk_km = float(UC_VOC_TRK) * factor_trk

    voc = vmt_car * voc_car_km + vmt_trk * voc_trk_km

    # ---- 3. Travel-time cost (TTC) with speed = f(IRI) ---------------------
    v_car_kmh = _uc_speed_from_iri(
        float(iri),
        float(UC_FFS_KPH),
        float(UC_IRI0),
        float(UC_K_KPH_PER_IRI),
        float(UC_MIN_SPEED),
    )
    # trucks use same speed as cars
    v_trk_kmh = v_car_kmh

    # hours per day = veh-km / (km/h)
    hours_car = vmt_car / max(1e-9, v_car_kmh)
    hours_trk = vmt_trk / max(1e-9, v_trk_kmh)

    # cars: passengers * VOT_car
    # trucks: driver only (UC_OCC_TRK usually = 1)
    ttc = (
        hours_car * float(UC_OCC_CAR) * float(UC_VOT_CAR)
        + hours_trk * float(UC_OCC_TRK) * float(UC_VOT_TRK)
    )

    return float(voc), float(ttc)

def _uc_add_window_cost(iri_series_window, year_start, year_end_inclusive):
    global user_cost_by_year
    y0 = int(year_start)
    span = int(year_end_inclusive) - y0 + 1
    for i in range(span):
        iri = float(iri_series_window[i])
        voc_d, ttc_d = _uc_daily_voc_ttc(iri)
        annual = 365.0 * (voc_d + ttc_d) * UC_SCALE
        y = y0 + i
        user_cost_by_year[y] = user_cost_by_year.get(y, 0.0) + annual

def rsi_pick(*items, design_life=None):
    """
    Return the earliest FIRST-CROSSING year across inputs.
    """
    candidates = []
    for x in items:
        if isinstance(x, (list, tuple)):
            last_ok = max([0] + [int(y) for y in x])
            first_cross = last_ok + 1
        else:
            first_cross = int(x)
        candidates.append(first_cross)

    if not candidates:
        return int(design_life or 0)

    vmin = min(candidates)
    if design_life is not None:
        vmin = min(vmin, int(design_life))
    return int(vmin)

def _segment_xy(series_or_slice, x0, x1, norm_thr):
    """
    Build (x, y) for plotting a segment [x0, x1] from an aligned series_or_slice.
    If the slice is shorter than requested, clamp to what's available so lengths match.
    """
    x0 = int(x0); x1 = int(x1)
    if x1 < x0:
        return np.array([]), np.array([])

    L_req  = x1 - x0 + 1                 # requested points
    arr    = np.asarray(series_or_slice, float)
    L_have = int(arr.shape[0])           # available points in the slice
    L      = min(L_req, L_have)          # clamp

    if L <= 0:
        return np.array([]), np.array([])

    seg = arr[:L]                        # exactly L points
    xr  = np.arange(x0, x0 + L)          # x0 .. x0+L-1
    y   = 1.0 - (seg / float(norm_thr))

    if xr.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch in _segment_xy: xr={xr.shape[0]} seg={seg.shape[0]} x0={x0} x1={x1}")

    return xr, y

def _safe_float(v):
    """Return float(v) or None if it can't be made a finite float."""
    try:
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None

def _area_key(r):
    """Key function for max(): returns -inf for missing/invalid area."""
    af = _safe_float(r.get('area', None))
    return af if af is not None else float('-inf')


# In[1]:


def function1f():
    # ------- globals -------
    global sn1, p, NPV, l, pre_pre, previous
    global RSI_preservation
    global area_IRI, area_fatigue, area_rutting
    global pre_IRI1, pre_fatigue1, pre_rutting1
    global rehab_IRI2, rehab_fatigue2, rehab_rutting2
    global recon_IRI3, recon_fatigue3, recon_rutting3
    global agency_cost_by_year

    cand = ac_thick * CO_A + TH_B * CO_B * DCO_B + TH_SUB * CO_SUB * DCO_SUB
    if not np.isnan(cand):
        sn1 = min(sn1, cand)

    v = ['RSI_Maintenance', RSI_preservation]
    history = globals().get("history", [])
    history.append(v)
    globals()["history"] = history

    # Calculating NPV using real rate
    real_rate = (1 + float(Interest_rate)) / (1 + float(Inflation_rate)) - 1
    NPV += float(Maintenance_Cost) * (1 + real_rate) ** (-int(RSI_preservation))
    
    agency_cost_by_year = globals().get("agency_cost_by_year", {})
    agency_cost_by_year[int(RSI_preservation)] = agency_cost_by_year.get(int(RSI_preservation), 0.0) + float(Maintenance_Cost)
   
    # --- plotting cosmetics ---
    plt.xlim([0,int(Design_life)])
    plt.ylim([0,1])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 11 
    
    l = l + 1
    
    if l == 1:
        x0, x1 = 0, int(RSI_preservation)
        src_IRI = IRI[x0:x1+1]
        src_fa  = fatigue[x0:x1+1]
        src_ru  = rutting[x0:x1+1]
        
        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)
        
        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return
        
        xr = xr1[:L]; y1 = y1[:L]; y2 = y2[:L]; y3 = y3[:L]
        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')
        
        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))
        
        x1_eff = int(xr[-1])
        area_per_year_final1 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)

        previous = [int(RSI_preservation), 1]
        pre_pre = [0, 0]
        
        # ---------- pre_IRI ----------
        miri_age = int(RSI_preservation)
        miri = 0.8 * IRI[miri_age]  # pre_pre[0] == 0 in this branch
        
        pre_IRI = iri_series(
            a_IRI=a_IRI, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT, AT=ac_thick,
            miri_age=miri_age, miri=miri, AGE=AGE, INI_IRI=miri
        )
        
        # ---------- pre_fatigue ----------
        if RSI_preservation < fca:
            first_crack_age = fca - RSI_preservation
        else: 
            first_crack_age = 1
            
        pfatigue = fatigue_series(
            a2_fatigue=a2_fatigue, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
            AGE=AGE, first_crack_age=first_crack_age
        )

        pre_IRI1      = [np.round(num, 4) for num in pre_IRI[int(previous[0]):int(Design_life) + 1]]
        pre_fatigue1  = [np.round(num, 4) for num in pfatigue[0:int(Design_life) + 1]]
        
        # ---------- pre_rutting ----------
        prutting = rutting_series(
            a_rutting=a_rutting, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
            AGE=AGE, beta=beta, gamma=gama
        )
        
        baseline = rutting[previous[0]] - 0.5 * (rutting[previous[0]] - rutting[0])
        y = _find_y(prutting, baseline)
        pre_rutting1 = [np.round(num, 4) for num in prutting[y:int(Design_life) + 1]]

    if l > 1:
        if previous[1] == 1:
            src_IRI, src_fa, src_ru = pre_IRI1, pre_fatigue1, pre_rutting1
            pre_code = 1
        elif previous[1] == 2:
            src_IRI, src_fa, src_ru = rehab_IRI2, rehab_fatigue2, rehab_rutting2
            pre_code = 2
        else:  # previous[1] == 3
            src_IRI, src_fa, src_ru = recon_IRI3, recon_fatigue3, recon_rutting3
            pre_code = 3

        x0, x1 = int(previous[0]), int(RSI_preservation) 
        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)
        
        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return
        
        xr = xr1[:L]; y1 = y1[:L]; y2 = y2[:L]; y3 = y3[:L]
        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')
        
        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))
        
        x1_eff = int(xr[-1])
        area_per_year_final1 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)

        pre_pre = [pre_code, previous[0]]
        previous = [int(RSI_preservation), 1]

        # ---------- pre_IRI ----------
        miri_age = int(RSI_preservation)
        if pre_pre[0] == 1:
            miri_src = pre_IRI1
        elif pre_pre[0] == 2:
            miri_src = rehab_IRI2
        elif pre_pre[0] == 3:
            miri_src = recon_IRI3
        else:
            miri_src = IRI

        j = int(previous[0] - pre_pre[1])
        if j < 0:
            j = 0
        elif j >= len(miri_src):
            j = len(miri_src) - 1
        miri = 0.8 * miri_src[j]

        pre_IRI = iri_series(
            a_IRI=a_IRI, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT, AT=ac_thick,
            miri_age=miri_age, miri=miri, AGE=AGE, INI_IRI=miri
        )
        
        # ---------- pre_fatigue ----------
        if RSI_preservation < fca:
            first_crack_age = fca - RSI_preservation
        else: 
            first_crack_age = 1
            
        pfatigue = fatigue_series(
            a2_fatigue=a2_fatigue, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
            AGE=AGE, first_crack_age=first_crack_age if RSI_preservation < fca else 1
        )
        
        prepre_IRI1 = pre_IRI1
        pre_IRI1     = [np.round(num, 4) for num in pre_IRI[int(previous[0]):int(Design_life) + 1]]
        pre_fatigue1 = [np.round(num, 4) for num in pfatigue[0:int(Design_life) + 1]]

        # ---------- pre_rutting ----------
        prutting = rutting_series(
            a_rutting=a_rutting, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
            AGE=AGE, beta=beta, gamma=gama
        )

        # set baseline according to pre_pre code
        if pre_pre[0] == 0:
            baseline = rutting[previous[0]] - 0.5 * (rutting[previous[0]] - rutting[0])
        elif pre_pre[0] == 1:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(pre_rutting1) - 1))
            baseline = pre_rutting1[j] - 0.5 * (pre_rutting1[j] - rutting[0])
        elif pre_pre[0] == 2:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(rehab_rutting2) - 1))
            baseline = rehab_rutting2[j] - 0.5 * (rehab_rutting2[j] - rutting[0])
        else:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(recon_rutting3) - 1))
            baseline = recon_rutting3[j] - 0.5 * (recon_rutting3[j] - rutting[0])

        y = _find_y(prutting, baseline)
        pre_rutting1 = [np.round(num, 4) for num in prutting[y:int(Design_life) + 1]]
        
    # ---------------- legend & grid ----------------
    plt.legend(['IRI', 'Fatigue', 'Rutting'])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    if int(RSI_preservation) == int(Design_life):
        return

    # --------- build preservation trajectory from RSI_preservation onward ---
    miri_age = int(RSI_preservation)
    if l > 1:
        # choose the same source as above
        if pre_pre[0] == 1:
            miri_src = prepre_IRI1
        elif pre_pre[0] == 2:
            miri_src = rehab_IRI2
        elif pre_pre[0] == 3:
            miri_src = recon_IRI3
        else:
            miri_src = IRI

        j = int(previous[0] - pre_pre[1])
        j = max(0, min(j, len(miri_src) - 1))
        miri = 0.8 * miri_src[j]
    else:
        miri = 0.8 * IRI[int(RSI_preservation)]

    pre_IRI_function = iri_series(
        a_IRI=a_IRI, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT, AT=ac_thick,
        miri_age=miri_age, miri=miri, AGE=AGE, INI_IRI=miri
    )
        
    # preservation series from RSI_preservation to end
    preservation_IRI      = pre_IRI_function[int(RSI_preservation):int(Design_life) + 1]
    preservation_fatigue  = pfatigue[0:int(Design_life) - int(RSI_preservation) + 1]

    # baseline for rutting
    if l == 1:
        baseline = rutting[previous[0]] - 0.5 * (rutting[previous[0]] - rutting[0])
    else:
        if pre_pre[0] == 1:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(pre_rutting1) - 1))
            baseline = pre_rutting1[j] - 0.5 * (pre_rutting1[j] - rutting[0])
        elif pre_pre[0] == 2:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(rehab_rutting2) - 1))
            baseline = rehab_rutting2[j] - 0.5 * (rehab_rutting2[j] - rutting[0])
        elif pre_pre[0] == 3:
            j = int(previous[0] - pre_pre[1] + 1)
            j = max(0, min(j, len(recon_rutting3) - 1))
            baseline = recon_rutting3[j] - 0.5 * (recon_rutting3[j] - rutting[0])
        else:
            baseline = rutting[previous[0]] - 0.5 * (rutting[previous[0]] - rutting[0])

    prutting = rutting_series(
        a_rutting=a_rutting, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AGE=AGE, beta=beta, gamma=gama
    )
    
    y = _find_y(prutting, baseline)
    preservation_rutting = prutting[y:int(Design_life) - int(RSI_preservation) + y + 1]

    preservation = np.array([preservation_IRI, preservation_fatigue, preservation_rutting], dtype=object)
    new_preservation = lambda y_, x_: preservation[y_, x_]

    # ---------------- re-create the three RSI lists ----------------
    IRI_thr = [IRI_preservation_threshold, IRI_rehabilitation_threshold, IRI_reconstruction_threshold]
    FAT_thr = [fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold]
    RUT_thr = [rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold]
    
    iri_pres, iri_rehab, iri_recon = rsi_lists_from_series(preservation[0], IRI_thr, RSI_preservation, Design_life)
    fat_pres, fat_rehab, fat_recon = rsi_lists_from_series(preservation[1], FAT_thr, RSI_preservation, Design_life)
    rut_pres, rut_rehab, rut_recon = rsi_lists_from_series(preservation[2], RUT_thr, RSI_preservation, Design_life)
    
    RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction = iri_pres, iri_rehab, iri_recon
    RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction = fat_pres, fat_rehab, fat_recon
    RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction = rut_pres, rut_rehab, rut_recon

    # ============== PATCH: correct combined RSI aggregation =================
    RSI_preservation   = rsi_pick(iri_pres, rut_pres, fat_pres)
    RSI_rehabilitation = rsi_pick(iri_rehab, rut_rehab, fat_rehab)
    RSI_reconstruction = rsi_pick(iri_recon, rut_recon, fat_recon)
    
    return {"area_per_year": area_per_year_final1, "label": v}


# In[8]:


def function2f():
    # ------- globals -------
    global sn1, ac_thick, p, NPV, l, pre_pre, previous
    global RSI_rehabilitation
    global area_IRI, area_fatigue, area_rutting
    global pre_IRI1, pre_fatigue1, pre_rutting1
    global rehab_IRI2, rehab_fatigue2, rehab_rutting2
    global recon_IRI3, recon_fatigue3, recon_rutting3
    global beta, gama
    global agency_cost_by_year

    sn1 = (ac_thick*CO_A + TH_B*CO_B*DCO_B + TH_SUB*CO_SUB*DCO_SUB + TH_OVER*CO_ASTL)

    v = ['RSI_rehabilitation', RSI_rehabilitation]
    history = globals().get("history", [])
    history.append(v); globals()["history"] = history

    real_rate = (1 + float(Interest_rate)) / (1 + float(Inflation_rate)) - 1
    NPV += float(Rehabilitation_Cost) * (1 + real_rate) ** (-int(RSI_rehabilitation))
    
    agency_cost_by_year = globals().get("agency_cost_by_year", {})
    agency_cost_by_year[int(RSI_rehabilitation)] = agency_cost_by_year.get(int(RSI_rehabilitation), 0.0) + float(Rehabilitation_Cost)

    # --- pick overlay-column IRI coeff ---
    if S3 == 'Fine':
        a_IRI = IRI_FI[2]
        a2_fatigue = Fatigue_FI[2]
        a_rutting = Rutting_FI[2]
    elif S3 == 'Coarse':
        a_IRI = IRI_CO[2]
        a2_fatigue = Fatigue_CO[2]
        a_rutting = Rutting_CO[2]
    else:  # 'Rock/Stone'
        a_IRI = IRI_RO[2]
        a2_fatigue =  Fatigue_RO[2]
        a_rutting =  Rutting_RO[2]
        
    # -------- beta, gama consistent with base-type mapping --------
    if S1 == 'Asphalt Treated Base':
        beta, gama = 0.331, -0.000228
    elif S1 == 'Dense Graded Aggregate Base':
        beta, gama = 0.225, -0.000266
    elif S1 == 'Lean Concrete Base':
        beta, gama = 0.0236, -0.00507
    elif S1 == 'Non Bituminous Treated Base':
        beta, gama = 0.108, 0.00120
    elif S1 == 'No Base':
        beta, gama = 1.13, 0.000283
    elif S1 == 'Permeable Asphalt Treated Base':
        beta, gama = 0.769, 0.000150

    # ---------- rehab_IRI (after rehabilitation) ----------
    ac_thick = AT + TH_OVER  # rehab augments AC thickness
    miri_age = int(RSI_rehabilitation)
    miri = min(0.789, IRI[0])

    rehab_IRI = iri_series(
        a_IRI=a_IRI, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AT=ac_thick, miri_age=miri_age, miri=miri, AGE=AGE, INI_IRI=miri
        )
    
    # ---------- rehab_fatigue ----------
    p_rehab_fatigue = [
        0.513 - 6.1e-2*(math.log10(ESAL)/sn1) - 1.15e-4*CI - 1.44e-4*FI,
        -math.log(0.7/(1-0.7)) + a1_fatigue + 0.917*(math.log10(ESAL)/sn1)
        - 1.12e-2*FT + 1.06e-4*FI - 2.21e-4*CI - 1.01e-3*RAIN
    ]
    roots = np.roots(p_rehab_fatigue)
    first_crack_age = float(roots[0]) if roots.size > 0 else 0.0
    if first_crack_age < 0: first_crack_age = 0.0

    rehab_fatigue = fatigue_series(
        a2_fatigue=a2_fatigue, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AGE=AGE, first_crack_age=first_crack_age
        )

    # ---------- rehab_rutting ----------
    rehab_rutting = rutting_series(
        a_rutting=a_rutting, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AGE=AGE, beta=beta, gamma=gama
        )

    # ---------- plotting frame ----------
    plt.xlim([0,int(Design_life)])
    plt.ylim([0,1])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 11 
    
    l = l + 1
    
    if l == 1:
        x0, x1 = 0, int(RSI_rehabilitation)
        src_IRI = IRI[x0:x1+1]
        src_fa  = fatigue[x0:x1+1]
        src_ru  = rutting[x0:x1+1]

        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)
        
        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return 
        
        xr = xr1[:L]
        y1 = y1[:L]
        y2 = y2[:L]
        y3 = y3[:L]
        
        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')
        
        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))
        
        x1_eff = int(xr[-1])
        area_per_year_final2 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)

        previous = [int(RSI_rehabilitation), 2]
        pre_pre = [0, 0]

        rehab_IRI2 = [np.round(num, 4) for num in rehab_IRI[int(previous[0]):int(Design_life) + 1]]
        rehab_fatigue2 = [np.round(num, 4) for num in rehab_fatigue[0:int(Design_life) + 1]]
        rehab_rutting2 = [np.round(num, 4) for num in rehab_rutting[0:int(Design_life) + 1]]

    if l > 1:
        if previous[1] == 1:
            src_IRI, src_fa, src_ru = pre_IRI1, pre_fatigue1, pre_rutting1
            pre_code = 1
        elif previous[1] == 2:
            src_IRI, src_fa, src_ru = rehab_IRI2, rehab_fatigue2, rehab_rutting2
            pre_code = 2
        else:
            src_IRI, src_fa, src_ru = recon_IRI3, recon_fatigue3, recon_rutting3
            pre_code = 3

        x0, x1 = int(previous[0]), int(RSI_rehabilitation)
        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)

        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return 
        
        xr = xr1[:L]
        y1 = y1[:L]
        y2 = y2[:L]
        y3 = y3[:L]

        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')

        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))

        x1_eff = int(xr[-1])
        area_per_year_final2 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)

        pre_pre = [pre_code, previous[0]]
        previous = [int(RSI_rehabilitation), 2]

        rehab_IRI2 = [np.round(num, 4) for num in rehab_IRI[int(previous[0]):int(Design_life) + 1]]
        rehab_fatigue2 = [np.round(num, 4) for num in rehab_fatigue[0:int(Design_life) + 1]]
        rehab_rutting2 = [np.round(num, 4) for num in rehab_rutting[0:int(Design_life) + 1]]

    # legend/grid
    plt.legend(['IRI', 'Fatigue', 'Rutting'])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    if int(RSI_rehabilitation) == int(Design_life):
        return

    rehabilitation_IRI      = rehab_IRI[int(RSI_rehabilitation):int(Design_life) + 1]
    rehabilitation_fatigue  = rehab_fatigue[0:int(Design_life) - int(RSI_rehabilitation) + 1]
    rehabilitation_rutting = rehab_rutting[0:int(Design_life) - int(RSI_rehabilitation) + 1]

    rehabilitation = np.array([rehabilitation_IRI,rehabilitation_fatigue,rehabilitation_rutting], dtype=object)
    new_rehabilitation = lambda y_, x_: rehabilitation[y_, x_]  # (kept for compatibility)

    # ----- recompute RSI lists on the rehab-forward series -----
    IRI_thr = [IRI_preservation_threshold, IRI_rehabilitation_threshold, IRI_reconstruction_threshold]
    FAT_thr = [fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold]
    RUT_thr = [rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold]
    
    iri_pres, iri_rehab, iri_recon = rsi_lists_from_series(rehabilitation[0], IRI_thr, RSI_rehabilitation, Design_life)
    fat_pres, fat_rehab, fat_recon = rsi_lists_from_series(rehabilitation[1], FAT_thr, RSI_rehabilitation, Design_life)
    rut_pres, rut_rehab, rut_recon = rsi_lists_from_series(rehabilitation[2], RUT_thr, RSI_rehabilitation, Design_life)
    
    RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction = iri_pres, iri_rehab, iri_recon
    RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction = fat_pres, fat_rehab, fat_recon
    RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction = rut_pres, rut_rehab, rut_recon

    # -------- combine RSIs (same rule as preservation) --------
    RSI_preservation   = rsi_pick(iri_pres, rut_pres, fat_pres)
    RSI_rehabilitation = rsi_pick(iri_rehab, rut_rehab, fat_rehab)
    RSI_reconstruction = rsi_pick(iri_recon, rut_recon, fat_recon)
    
    return {"area_per_year": area_per_year_final2, "label": v}


# In[9]:


def function3f():
    # ------- globals -------
    global area_IRI, area_fatigue, area_rutting
    global pre_IRI1, pre_fatigue1, pre_rutting1
    global rehab_IRI2, rehab_fatigue2, rehab_rutting2
    global recon_IRI3, recon_fatigue3, recon_rutting3
    global RSI_reconstruction
    global l, NPV, sn1, beta, gama, pre_pre, previous
    global agency_cost_by_year

    v = ['RSI_reconstruction', RSI_reconstruction]
    history = globals().get("history", [])
    history.append(v); globals()["history"] = history

    real_rate = (1 + float(Interest_rate)) / (1 + float(Inflation_rate)) - 1
    NPV += float(Reconstruction_Cost) * (1 + real_rate) ** (-int(RSI_reconstruction))
    
    agency_cost_by_year = globals().get("agency_cost_by_year", {})
    agency_cost_by_year[int(RSI_reconstruction)] = agency_cost_by_year.get(int(RSI_reconstruction), 0.0) + float(Reconstruction_Cost)

    # ---- choose the overlay-column coeffs, same as 2f ----
    if S3 == 'Fine':
        a2_fatigue = Fatigue_FI[2]; a_rutting = Rutting_FI[2]; a_IRI = IRI_FI[2]
    elif S3 == 'Coarse':
        a2_fatigue = Fatigue_CO[2]; a_rutting = Rutting_CO[2]; a_IRI = IRI_CO[2]
    else:  # 'Rock/Stone'
        a2_fatigue = Fatigue_RO[2]; a_rutting = Rutting_RO[2]; a_IRI = IRI_RO[2]

    # ---- base-type mapping aligned with wizard ----
    if S1 == 'Asphalt Treated Base':
        beta, gama = 0.331, -0.000228
    elif S1 == 'Dense Graded Aggregate Base':
        beta, gama = 0.225, -0.000266
    elif S1 == 'Lean Concrete Base':
        beta, gama = 0.0236, -0.00507
    elif S1 == 'Non Bituminous Treated Base':
        beta, gama = 0.108, 0.00120
    elif S1 == 'No Base':
        beta, gama = 1.13, 0.000283
    elif S1 == 'Permeable Asphalt Treated Base':
        beta, gama = 0.769, 0.000150

    # ---- reconstruction series ----
    miri_age = int(RSI_reconstruction)
    miri = min(0.789, float(IRI[0]))

    recon_IRI = iri_series(
        a_IRI=a_IRI, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT, AT=ac_thick,
        miri_age=miri_age, miri=miri, AGE=AGE, INI_IRI=miri
    )

    p_reconstruction_fatigue = [
        0.513 - 6.1e-2 * (math.log10(ESAL)/sn1) - 1.15e-4 * CI - 1.44e-4 * FI,
        -math.log(0.7/(1-0.7)) + a1_fatigue + 0.917*(math.log10(ESAL)/sn1)
        - 1.12e-2*FT + 1.06e-4*FI - 2.21e-4*CI - 1.01e-3*RAIN
    ]
    rts = np.roots(p_reconstruction_fatigue)
    first_crack_age = float(np.real(rts[0])) if rts.size > 0 else 0.0
    if first_crack_age < 0: first_crack_age = 0.0

    recon_fatigue = fatigue_series(
        a2_fatigue=a2_fatigue, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AGE=AGE, first_crack_age=first_crack_age
    )

    recon_rutting = rutting_series(
        a_rutting=a_rutting, ESAL=ESAL, SN=sn1, CI=CI, FI=FI, RAIN=RAIN, FT=FT,
        AGE=AGE, beta=beta, gamma=gama
    )

    # ---- plotting frame ----
    plt.xlim([0, int(Design_life)])
    plt.ylim([0, 1])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11

    l = l + 1
   
    if l == 1:
        x0, x1 = 0, int(RSI_reconstruction)
        src_IRI = IRI[x0:x1+1]
        src_fa  = fatigue[x0:x1+1]
        src_ru  = rutting[x0:x1+1]

        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)
        
        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return 
        
        xr = xr1[:L]
        y1 = y1[:L]
        y2 = y2[:L]
        y3 = y3[:L]
        
        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')
        
        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))
        
        x1_eff = int(xr[-1])
        area_per_year_final3 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)

        previous = [int(RSI_reconstruction), 3]
        pre_pre = [0, 0]

        recon_IRI3     = [np.round(num, 4) for num in recon_IRI[int(previous[0]):int(Design_life) + 1]]
        recon_fatigue3 = [np.round(num, 4) for num in recon_fatigue[0:int(Design_life) + 1]]
        recon_rutting3 = [np.round(num, 4) for num in recon_rutting[0:int(Design_life) + 1]]

    if l > 1:
        if previous[1] == 1:
            src_IRI, src_fa, src_ru = pre_IRI1,   pre_fatigue1,   pre_rutting1;   pre_code = 1
        elif previous[1] == 2:
            src_IRI, src_fa, src_ru = rehab_IRI2, rehab_fatigue2, rehab_rutting2; pre_code = 2
        else:
            src_IRI, src_fa, src_ru = recon_IRI3, recon_fatigue3, recon_rutting3; pre_code = 3

        x0, x1 = int(previous[0]), int(RSI_reconstruction) 
        xr1, y1 = _segment_xy(src_IRI, x0, x1, IRI_reconstruction_threshold)
        xr2, y2 = _segment_xy(src_fa,  x0, x1, fatigue_reconstruction_threshold)
        xr3, y3 = _segment_xy(src_ru,  x0, x1, rutting_reconstruction_threshold)
        
        L = min(len(xr1), len(xr2), len(xr3))
        if L <= 0:
            return 
        
        xr = xr1[:L]
        y1 = y1[:L]
        y2 = y2[:L]
        y3 = y3[:L]
        
        line1, = plt.plot(xr, y1, color='red',   linestyle='--', label='IRI')
        line2, = plt.plot(xr, y2, color='blue',  linestyle='-',  label='Fatigue')
        line3, = plt.plot(xr, y3, color='green', linestyle='-.', label='Rutting')
        
        area_IRI     += float(np.trapz(1.0 - y1, xr))
        area_fatigue += float(np.trapz(1.0 - y2, xr))
        area_rutting += float(np.trapz(1.0 - y3, xr))
        
        x1_eff = int(xr[-1])
        area_per_year_final3 = _windowed_area_per_year(src_IRI, src_fa, src_ru, x0, x1_eff)
        L = x1_eff - x0

        pre_pre = [pre_code, previous[0]]
        previous = [int(RSI_reconstruction), 3]

        recon_IRI3     = [np.round(num, 4) for num in recon_IRI[int(previous[0]):int(Design_life) + 1]]
        recon_fatigue3 = [np.round(num, 4) for num in recon_fatigue[0:int(Design_life) + 1]]
        recon_rutting3 = [np.round(num, 4) for num in recon_rutting[0:int(Design_life) + 1]]

    plt.legend(['IRI', 'Fatigue', 'Rutting'])
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    if int(RSI_reconstruction) == int(Design_life):
        return

    # ---- forward series from RSI_reconstruction ----
    reconstruction_IRI      = recon_IRI[int(RSI_reconstruction):int(Design_life) + 1]
    reconstruction_fatigue  = recon_fatigue[0:int(Design_life) - int(RSI_reconstruction) + 1]
    reconstruction_rutting  = recon_rutting[0:int(Design_life) - int(RSI_reconstruction) + 1]
    reconstruction = np.array([reconstruction_IRI, reconstruction_fatigue, reconstruction_rutting], dtype=object)

    # thresholds
    IRI_thr = [IRI_preservation_threshold, IRI_rehabilitation_threshold, IRI_reconstruction_threshold]
    FAT_thr = [fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold]
    RUT_thr = [rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold]

    iri_pres, iri_rehab, iri_recon = rsi_lists_from_series(reconstruction[0], IRI_thr, RSI_reconstruction, Design_life)
    fat_pres, fat_rehab, fat_recon = rsi_lists_from_series(reconstruction[1], FAT_thr, RSI_reconstruction, Design_life)
    rut_pres, rut_rehab, rut_recon = rsi_lists_from_series(reconstruction[2], RUT_thr, RSI_reconstruction, Design_life)

    RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction = iri_pres, iri_rehab, iri_recon
    RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction = fat_pres, fat_rehab, fat_recon
    RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction = rut_pres, rut_rehab, rut_recon

    # combine RSIs like elsewhere
    RSI_preservation   = rsi_pick(iri_pres, rut_pres, fat_pres)
    RSI_rehabilitation = rsi_pick(iri_rehab, rut_rehab, fat_rehab)
    RSI_reconstruction = rsi_pick(iri_recon, rut_recon, fat_recon)

    return {"area_per_year": area_per_year_final3, "label": v}


# In[10]:


ACTIONS = {1: function1f, 2: function2f, 3: function3f}

@contextlib.contextmanager
def _suppress_internal_legends():
    """Temporarily turn plt.legend into a no-op so step functions can't add legends."""
    old = plt.legend
    try:
        plt.legend = lambda *a, **k: None
        yield
    finally:
        plt.legend = old

plt.ioff()  # keep everything quiet / no interactive windows

# --- Salvage settings ---
SALVAGE_ENABLE = True
SALVAGE_USE_DYNAMIC_LIFE = True     # use RSI-based life computed after each treatment
# Fallback design lives (used only if a dynamic life can't be computed)
SALVAGE_LIFE_YRS = {
    "RSI_Maintenance":    5.0,
    "RSI_Rehabilitation": 12.0,
    "RSI_Reconstruction": 25.0,
}
# only major actions:
SALVAGE_ALLOW_KINDS = {"RSI_Rehabilitation", "RSI_Reconstruction"}  # exclude maintenance by default

_STATE_VARS = [
    # numeric accumulators
    'sn1','ac_thick','NPV','l','area_IRI','area_fatigue','area_rutting',
    # series that become the new baselines after a step
    'pre_IRI1','pre_fatigue1','pre_rutting1',
    'rehab_IRI2','rehab_fatigue2','rehab_rutting2',
    'recon_IRI3','recon_fatigue3','recon_rutting3',
    # RSI progress lists & combined picks
    'RSI_IRI_preservation','RSI_IRI_rehabilitation','RSI_IRI_reconstruction',
    'RSI_rutting_preservation','RSI_rutting_rehabilitation','RSI_rutting_reconstruction',
    'RSI_fatigue_preservation','RSI_fatigue_rehabilitation','RSI_fatigue_reconstruction',
    'RSI_preservation','RSI_rehabilitation','RSI_reconstruction',
    # control state
    'previous','pre_pre','history',
    'beta','gama','fca',
    'agency_cost_by_year', 'user_cost_by_year',
]

# in-memory registry to avoid saving duplicates by area
_AREA_REGISTRY: dict[float, str] = {}

def _ir_window_and_bounds(ir_series, prev_code, prev0_abs, y0, y1):
    """
    Return (seg_ir, y0, y1_adj) where seg_ir is a 1D array whose length exactly matches
    the window [y0..y1_adj]. If the requested [y0..y1] overruns the series, y1_adj is trimmed.
    prev_code: 0 for do-nothing baseline (absolute indexing),
               1/2/3 for treatment-aligned series (index 0 corresponds to abs year prev0_abs).
    """
    y0 = int(y0); y1 = int(y1)
    if y1 < y0:
        return np.asarray([], float), y0, y0 - 1

    n_req = y1 - y0 + 1
    if prev_code == 0:
        seg = np.asarray(ir_series[y0 : y0 + n_req], float)
    else:
        offset = int(y0 - int(prev0_abs))
        seg = np.asarray(ir_series[offset : offset + n_req], float)

    y1_adj = y0 + int(seg.shape[0]) - 1
    return seg, y0, y1_adj

def _series_for_prev():
    """Return (IRI, fatigue, rutting) for the current previous[1] code."""
    if previous[1] == 1:
        return pre_IRI1,   pre_fatigue1,   pre_rutting1
    if previous[1] == 2:
        return rehab_IRI2, rehab_fatigue2, rehab_rutting2
    if previous[1] == 3:
        return recon_IRI3, recon_fatigue3, recon_rutting3
    # "do nothing" baseline:
    return IRI, fatigue, rutting

def _plot_tail_and_area():
    """
    Draw the last segment from previous[0]..Design_life for the current baseline series,
    update area accumulators, and return (line1, line2, line3, tail_per_year).
    """
    global area_IRI, area_fatigue, area_rutting

    ir, fa, ru = _series_for_prev()
    start_abs = int(previous[0]) if previous[1] != 0 else 0
    end_abs   = int(Design_life)
    L = end_abs - start_abs
    if L <= 0:
        return None, None, None, []

    # Align arrays so index 0 is the start of this tail:
    if previous[1] == 0:  # do-nothing baseline
        seg_ir = np.asarray(ir[start_abs:end_abs+1], float)
        seg_fa = np.asarray(fa[start_abs:end_abs+1], float)
        seg_ru = np.asarray(ru[start_abs:end_abs+1], float)
    else:
        # pre_/rehab_/recon_ series are already aligned at previous[0]
        seg_ir = np.asarray(ir[:L+1], float)
        seg_fa = np.asarray(fa[:L+1], float)
        seg_ru = np.asarray(ru[:L+1], float)

    x = np.arange(start_abs, end_abs + 1)

    y1 = 1.0 - (seg_ir / float(IRI_reconstruction_threshold))
    y2 = 1.0 - (seg_fa / float(fatigue_reconstruction_threshold))
    y3 = 1.0 - (seg_ru / float(rutting_reconstruction_threshold))

    line1, = plt.plot(x, y1, linestyle='--', color='r', label='IRI')
    line2, = plt.plot(x, y2, linestyle='-', color='b', label='Fatigue')
    line3, = plt.plot(x, y3, linestyle='-.', color='g', label='Rutting')

    # Accumulate areas on normalized series
    area_IRI     += float(np.trapz(1.0 - y1, x))
    area_fatigue += float(np.trapz(1.0 - y2, x))
    area_rutting += float(np.trapz(1.0 - y3, x))

    # Per-year tail (deterministic)
    tail_per_year = _windowed_area_per_year(seg_ir, seg_fa, seg_ru, 0, L)
    return line1, line2, line3, tail_per_year

def _compute_tail_and_area_only():
    """
    Compute the final tail's per-year area and its sum without plotting
    and without mutating accumulators. Works for both do-nothing and
    treatment-aligned series.
    """
    ir, fa, ru = _series_for_prev()
    start_abs = int(previous[0]) if previous[1] != 0 else 0
    end_abs   = int(Design_life)
    L = end_abs - start_abs
    if L <= 0:
        return [], 0.0

    if previous[1] == 0:  # do-nothing baseline: absolute-year indexing
        seg_ir = np.asarray(ir[start_abs:end_abs+1], float)
        seg_fa = np.asarray(fa[start_abs:end_abs+1], float)
        seg_ru = np.asarray(ru[start_abs:end_abs+1], float)
    else:                 # treatment-aligned: index from 0..L
        seg_ir = np.asarray(ir[:L+1], float)
        seg_fa = np.asarray(fa[:L+1], float)
        seg_ru = np.asarray(ru[:L+1], float)

    tail_per_year = _windowed_area_per_year(seg_ir, seg_fa, seg_ru, 0, L)
    return tail_per_year, float(sum(tail_per_year))

def _save_state():
    return {k: deepcopy(globals().get(k)) for k in _STATE_VARS}

def _restore_state(st):
    for k, v in st.items():
        globals()[k] = deepcopy(v)

def _reset_iteration_baseline():
    global sn1, ac_thick, p, NPV, l, area_IRI, area_fatigue, area_rutting
    global pre_IRI1, pre_fatigue1, pre_rutting1
    global rehab_IRI2, rehab_fatigue2, rehab_rutting2
    global recon_IRI3, recon_fatigue3, recon_rutting3
    global previous, pre_pre
    # globals for RSI rebuild
    global RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction
    global RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction
    global RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction
    global RSI_preservation, RSI_rehabilitation, RSI_reconstruction, history

    sn1      = float(SN)
    ac_thick = float(AT)
    p = [0, 0]
    NPV = 0
    l = 0
    area_IRI = area_fatigue = area_rutting = 0.0
    pre_IRI1 = pre_fatigue1 = pre_rutting1 = 0
    rehab_IRI2 = rehab_fatigue2 = rehab_rutting2 = 0
    recon_IRI3 = recon_fatigue3 = recon_rutting3 = 0
    previous = [0, 0]
    pre_pre  = [0, 0]
    history = []

    # Rebuild the RSI lists from the original baseline series
    RSI_IRI_preservation, RSI_IRI_rehabilitation, RSI_IRI_reconstruction = rsi_three_stage(
        IRI, IRI_preservation_threshold, IRI_rehabilitation_threshold, IRI_reconstruction_threshold, Design_life
    )
    RSI_rutting_preservation, RSI_rutting_rehabilitation, RSI_rutting_reconstruction = rsi_three_stage(
        rutting, rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold, Design_life
    )
    RSI_fatigue_preservation, RSI_fatigue_rehabilitation, RSI_fatigue_reconstruction = rsi_three_stage(
        fatigue, fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold, Design_life
    )

    # Combined picks
    RSI_preservation   = rsi_pick(RSI_IRI_preservation,   RSI_rutting_preservation,   RSI_fatigue_preservation)
    RSI_rehabilitation = rsi_pick(RSI_IRI_rehabilitation, RSI_rutting_rehabilitation, RSI_fatigue_rehabilitation)
    RSI_reconstruction = rsi_pick(RSI_IRI_reconstruction, RSI_rutting_reconstruction, RSI_fatigue_reconstruction)

@contextlib.contextmanager
def _fresh_fig():
    fig = plt.figure()
    try:
        yield fig
    finally:
        plt.close(fig)

def finalize_costs(base_year: int,
                   end_year: int,
                   real_rate: float,
                   agency_cost_by_year: dict,
                   user_cost_by_year: dict,
                   fallback_agency_npv: float = None):
    """
    Build a summary table and NPVs from annual agency & user costs.
    """
    years = list(range(int(base_year), int(end_year) + 1))
    df = pd.DataFrame({"Year": years})
    df["Agency($/yr)"] = df["Year"].map(lambda y: float(agency_cost_by_year.get(y, 0.0)))
    df["User($/yr)"]   = df["Year"].map(lambda y: float(user_cost_by_year.get(y, 0.0)))
    df["Total($/yr)"]  = df["Agency($/yr)"] + df["User($/yr)"]

    # Discount factors relative to base_year
    df["DF"] = 1.0 / ((1.0 + float(real_rate)) ** (df["Year"] - int(base_year)))
    df["Agency_PV"] = df["Agency($/yr)"] * df["DF"]
    df["User_PV"]   = df["User($/yr)"]   * df["DF"]
    df["Total_PV"]  = df["Total($/yr)"]  * df["DF"]

    cost_agency = float(df["Agency_PV"].sum())
    cost_user   = float(df["User_PV"].sum())
    cost_total  = float(df["Total_PV"].sum())

    summary = {
        "Agency_Cost($)": cost_agency,
        "User_Cost($)":   cost_user,
        "Total_Cost($)":  cost_total
    }
    if fallback_agency_npv is not None:
        summary["Agency_Cost_fallback($)"] = float(fallback_agency_npv)

    return df, summary
    
def _simulate_sequence(actions, render_terminal=True):
    _reset_iteration_baseline()

    df_costs = None
    seq_str = None

    # Record initial RSIs (target intervals for RSI-based salvage)
    init_rsis = {
        1: float(RSI_preservation),
        2: float(RSI_rehabilitation),
        3: float(RSI_reconstruction),
    }

    # reset cost logs for this run
    global agency_cost_by_year, user_cost_by_year
    agency_cost_by_year = {}
    user_cost_by_year   = {}
    st0 = _save_state()

    final_path = None
    area = None

    with _fresh_fig():
        executed_actions = []
        results = []
        area_chunks = []
        
        with _suppress_internal_legends():
            for a in actions:
                if a == 0:
                    if RSI_preservation   < Design_life: globals()['RSI_preservation']   = Design_life
                    if RSI_rehabilitation < Design_life: globals()['RSI_rehabilitation'] = Design_life
                    if RSI_reconstruction < Design_life: globals()['RSI_reconstruction'] = Design_life
                    break

                # 1) snapshot full state before trying the step
                st_before = _save_state()
                prev0_before = int(previous[0]) if isinstance(previous, (list, tuple)) else -1
                
                # 2) try the step
                res = ACTIONS[a]()
                prev0_after  = int(previous[0]) if isinstance(previous, (list, tuple)) else -1
                
                # never apply a treatment that lands at/after end-of-life
                if prev0_after >= int(Design_life):
                    _restore_state(st_before)   
                    break                       

                # 3) accept only if it truly advanced the clock and returned a non-empty per-year area
                progressed = prev0_after > prev0_before
                seg = res.get("area_per_year", []) if isinstance(res, dict) else []
                lab = res.get("label") if isinstance(res, dict) else None
                seg_len  = len(seg)
                applied  = progressed and (seg_len > 0)
        
                if applied:
                    area_chunks.append(seg)    
                    results.append(lab)         
                    executed_actions.append(a)
                else:
                    # 4) rollback all global side-effects from this non-applied step
                    _restore_state(st_before)
                    break
        
                # original guards
                if (RSI_preservation   < previous[0]) or \
                   (RSI_rehabilitation < previous[0]) or \
                   (RSI_reconstruction < previous[0]):
                    break
                if (RSI_preservation   == Design_life) and \
                   (RSI_rehabilitation == Design_life) and \
                   (RSI_reconstruction == Design_life):
                    break
            # ---- end of restored loop ----

        # ---------- add ALL user-cost windows ----------
        real_rate = (1 + float(Interest_rate)) / (1 + float(Inflation_rate)) - 1
        end_abs   = int(Design_life)
        
        # Collect accepted action years in order, e.g. [6, 11]
        action_years = []
        pre_codes    = []  
        pre_series   = []  
        
        # Keep a cursor of where the current window starts.
        y_cursor = 0
        
        # To build windows correctly, need the baseline before each accepted action.
        _state_snapshot = _save_state()        # save current post-loop state
        _restore_state(st0)                    # restore to start of run (before any steps)
        
        for a in actions:
            if a == 0:
                break
            # try the step and see if it truly applied
            st_before = _save_state()
            ir_pre, _, _ = _series_for_prev()               
            prev_code_before = st_before['previous'][1]
        
            res = ACTIONS[a]()                              # attempt
            progressed = int(previous[0]) > int(st_before['previous'][0])
            seg_len = len(res.get("area_per_year", [])) if isinstance(res, dict) else 0
            applied = progressed and (seg_len > 0)
        
            if applied:
                lab = res.get("label") if isinstance(res, dict) else None
                if lab and isinstance(lab, (list, tuple)) and len(lab) == 2:
                    y_act = int(lab[1])
        
                    # ---- window before the action ----
                    y0, y1 = int(y_cursor), int(y_act - 1)
                    if y1 >= y0:
                        seg_ir, y0_eff, y1_eff = _ir_window_and_bounds(ir_pre, prev_code_before, st_before['previous'][0], y0, y1)
                        if seg_ir.size:
                            _uc_add_window_cost(seg_ir, y0_eff, y1_eff)
        
                    # advance cursor to the action year
                    y_cursor = int(y_act)
        
                    # (keep for clarity/inspection if needed)
                    action_years.append(y_act)
                    pre_codes.append(prev_code_before)
                    pre_series.append(ir_pre)
        
                else:
                    # no label → can't place the window; abort adding user windows here
                    pass
            else:
                # not applied → roll back and stop replaying
                _restore_state(st_before)
                break
        
        # after replay finished
        uc_from_replay = deepcopy(user_cost_by_year)
        
        # bring back the real post-loop state
        _restore_state(_state_snapshot)
        
        # merge replayed windows + keep anything already in the dict
        for y, v in uc_from_replay.items():
            user_cost_by_year[y] = user_cost_by_year.get(y, 0.0) + v
        
        # ---- Tail window: from y_cursor to end_abs on the FINAL baseline (after last action) ----
        ir_tail, _, _ = _series_for_prev()
        y0, y1 = int(y_cursor), int(end_abs)
        seg_ir, y0_eff, y1_eff = _ir_window_and_bounds(ir_tail, previous[1], previous[0], y0, y1)
        if seg_ir.size:
            _uc_add_window_cost(seg_ir, y0_eff, y1_eff)
        # ---------- end NEW user-cost windows ----------

        # compute alive / terminal
        alive = []
        if RSI_preservation   < Design_life: alive.append(1)
        if RSI_rehabilitation < Design_life: alive.append(2)
        if RSI_reconstruction < Design_life: alive.append(3)
        terminal = (not alive) or (actions and actions[-1] == 0) or \
                   ((RSI_preservation < previous[0]) or (RSI_rehabilitation < previous[0]) or (RSI_reconstruction < previous[0]))
        
        # >>> PLACEHOLDERS so 'summary' fields exist even on probe runs
        cost_agency = 0.0
        cost_user   = 0.0
        cost_total  = 0.0

        if terminal:
            # Always compute tail/per-year/area (no plotting, no side-effects)
            tail, tail_area = _compute_tail_and_area_only()
            per_year = [e for chunk in area_chunks for e in chunk] + tail
            area = round(sum(per_year), 2)
            
        if render_terminal:
            # frame
            plt.title(results, loc='center', wrap=True)
            plt.rcParams['axes.titlesize'] = 10
            plt.xlabel('Age(year)')
            plt.ylabel('Performance of IRI & Fatigue Cracking & Rutting')
            plt.xlim([0, int(Design_life)])
            plt.ylim([0, 1])
            plt.rcParams["font.family"] = "Times New Roman"
            plt.rcParams['font.size'] = 11

            # draw last segment and compute area
            line1, line2, line3, tail = _plot_tail_and_area()

            # recompute combined RSIs BEFORE salvage (final RSIs at horizon)
            globals()['RSI_preservation']   = rsi_pick(RSI_IRI_preservation,   RSI_rutting_preservation,   RSI_fatigue_preservation)
            globals()['RSI_rehabilitation'] = rsi_pick(RSI_IRI_rehabilitation, RSI_rutting_rehabilitation, RSI_fatigue_rehabilitation)
            globals()['RSI_reconstruction'] = rsi_pick(RSI_IRI_reconstruction, RSI_rutting_reconstruction, RSI_fatigue_reconstruction)
            
            # ---------------- SALVAGE CREDIT (per lane-km) ----------------
            if SALVAGE_ENABLE and executed_actions:
                # last action kind: 1=maintenance, 2=rehabilitation, 3=reconstruction
                last_kind = int(executed_actions[-1])
                kind_name = {1: "RSI_Maintenance",
                             2: "RSI_Rehabilitation",
                             3: "RSI_Reconstruction"}[last_kind]
            
                # only credit the kinds allowed by your policy
                if kind_name in SALVAGE_ALLOW_KINDS:
                    unit_cost = {1: float(Maintenance_Cost),
                                 2: float(Rehabilitation_Cost),
                                 3: float(Reconstruction_Cost)}[last_kind]
            
                    # denominator for salvage fraction
                    if SALVAGE_USE_DYNAMIC_LIFE:
                        # use the initial RSI captured at the start of this run
                        target = max(1e-9, float(init_rsis[last_kind]))
                    else:
                        # use fixed service lives table (string keys)
                        target = max(1e-9, float(SALVAGE_LIFE_YRS[kind_name]))
            
                    # remaining interval at the end of the analysis horizon
                    remaining = {1: float(RSI_preservation),
                                 2: float(RSI_rehabilitation),
                                 3: float(RSI_reconstruction)}[last_kind]
                    remaining = max(0.0, remaining)
            
                    # fraction 0..1 and **per lane-km** salvage credit
                    frac = max(0.0, min(remaining / target, 1.0))
                    salvage_credit_per_lkm = unit_cost * frac
            
                    # book salvage as a negative agency cost in the final year (per lane-km)
                    end_y = int(Design_life)
                    agency_cost_by_year[end_y] = agency_cost_by_year.get(end_y, 0.0) - salvage_credit_per_lkm
                    # ---------------- end salvage ----------------
        
            # --- Construction cost as a year-0 agency cost (per lane-km) ---
            agency_cost_by_year[0] = agency_cost_by_year.get(0, 0.0) + float(Construction_Cost)

            df_costs, summary = finalize_costs(
                base_year=0,
                end_year=int(Design_life),
                real_rate=real_rate,
                agency_cost_by_year=agency_cost_by_year,
                user_cost_by_year=user_cost_by_year,
                fallback_agency_npv=float(NPV)
            )
            
            # recompute combined RSI picks
            globals()['RSI_preservation']   = rsi_pick(RSI_IRI_preservation,   RSI_rutting_preservation,   RSI_fatigue_preservation)
            globals()['RSI_rehabilitation'] = rsi_pick(RSI_IRI_rehabilitation, RSI_rutting_rehabilitation, RSI_fatigue_rehabilitation)
            globals()['RSI_reconstruction'] = rsi_pick(RSI_IRI_reconstruction, RSI_rutting_reconstruction, RSI_fatigue_reconstruction)

            flat = [e for chunk in area_chunks for e in chunk] + tail
            area = round(sum(flat), 2)

            # Flatten per-year area for the entire run
            per_year = [e for chunk in area_chunks for e in chunk] + tail
            per_year = [round(x, 2) for x in per_year] 

            proxies = [
                Line2D([0],[0], color='r', linestyle='--'),
                Line2D([0],[0], color='b', linestyle='-'),
                Line2D([0],[0], color='g', linestyle='-.'),
            ]
            plt.legend(proxies, ['IRI', 'Fatigue', 'Rutting'])
            
            seq_str = "-".join(map(str, executed_actions)) or "∅"
            if area not in _AREA_REGISTRY:
                final_path = out_dir / f"area={area}__seq={seq_str}.pdf"
                plt.savefig(final_path)
                _AREA_REGISTRY[area] = final_path
            else:
                final_path = _AREA_REGISTRY[area]

            cost_agency = summary["Agency_Cost($)"]
            cost_user   = summary["User_Cost($)"]
            cost_total  = summary["Total_Cost($)"]

    # Turn executed actions into a tuple
    seq_tuple = tuple(executed_actions)

    if terminal and area is None:
        raise RuntimeError("Invariant broken: terminal run must have an area.")
    
    # signature and return
    sig = (
        globals()['RSI_preservation'],
        globals()['RSI_rehabilitation'],
        globals()['RSI_reconstruction'],
        tuple(globals()['previous']) if isinstance(globals().get('previous'), (list, tuple)) else (None, None),
    )

    out = {
        'terminal': terminal,
        'alive': alive,
        'area': area,
        'filename': final_path,
        'seq': seq_tuple,
        'per_year': per_year if terminal and render_terminal else None,
        'labels': results if terminal else None,
        'npv': float(NPV),
        'cost_agency': cost_agency,
        'cost_user':   cost_user,
        'cost_total':  cost_total,
        'sig': sig, 
        'df_costs': df_costs,
    }

    _restore_state(st0)
    return out

def enumerate_all_sequences():
    """Silent DFS; saves exactly one PDF per unique area value."""
    global _AREA_REGISTRY
    _AREA_REGISTRY = {}
    results_index = []

    def dfs(prefix):
        base = _simulate_sequence(prefix, render_terminal=False)
        base_prev_year = base['sig'][3][0] if base['sig'][3] else -1

        # Probe each candidate action by actually simulating it
        alive = []
        for a in (1, 2, 3):
            nxt = _simulate_sequence(prefix + [a], render_terminal=False)
            nxt_prev_year = nxt['sig'][3][0] if nxt['sig'][3] else -1
            if (nxt_prev_year > base_prev_year) and (nxt_prev_year < int(Design_life)):
                alive.append(a)

        # If nothing can progress further, this is terminal; render and record
        if not alive:
            final = _simulate_sequence(prefix, render_terminal=True)
            results_index.append(final)
            return

        # Explore all progressing actions, plus an explicit 'end-now' branch
        for a in alive + [0]:
            dfs(prefix + [a])

    dfs([])
    return results_index


# In[11]:


def rebuild_baseline_series_from_globals():
    """
    Build the *baseline* (do-nothing) distress series for the CURRENT section
    """
    g = globals()

    # --- pull inputs from globals (cast once to be safe) ---
    n        = int(g["Design_life"])
    ESAL     = float(g["ESAL"])
    SN       = float(g["SN"])
    AT       = float(g["AT"])
    FT       = float(g["FT"])
    FI       = float(g["FI"])
    CI       = float(g["CI"])
    RAIN     = float(g["RAIN"])
    AGE_IRI  = float(g["AGE_IRI"])
    INI_IRI  = float(g["INI_IRI"])

    a_IRI       = float(g["a_IRI"])
    a_rutting   = float(g["a_rutting"])
    a1_fatigue  = float(g["a1_fatigue"])
    a2_fatigue  = float(g["a2_fatigue"])
    beta        = float(g["beta"])
    gama        = float(g["gama"])

    # --- time axis ---
    AGE = np.arange(n + 1, dtype=int)
    g["AGE"] = AGE

    # ================= IRI =================
    IRI = []
    for k in range(len(AGE)):
        DELTA = round(
            a_IRI
            + 0.115 * (math.log10(ESAL)/SN)
            + 7.9e-3 * AGE_IRI
            - 4.33e-5 * CI
            + 2.28e-6 * FI
            + 5.9e-5 * RAIN
            + 2.21e-4 * FT
            + 8.59e-5 * AT
            - 5.39e-3 * (math.log10(ESAL)/SN) * AGE_IRI
            + 1.77e-6 * AGE_IRI * CI
            + 4.55e-6 * AGE_IRI * FI
            + 0.643 * INI_IRI
            - 2.4e-6 * AGE_IRI * RAIN
            - 1.09e-5 * AGE_IRI * FT
            - math.log(INI_IRI + 0.1),
            4
        )

        iri_k = round(
            math.exp(
                a_IRI - DELTA
                + 0.115 * (math.log10(ESAL)/SN)
                + 3.29e-2 * AGE[k]
                - 4.33e-5 * CI
                + 2.28e-6 * FI
                + 5.9e-5 * RAIN
                + 2.21e-4 * FT
                + 8.59e-5 * AT
                - 5.39e-3 * (math.log10(ESAL)/SN) * AGE[k]
                + 1.77e-6 * AGE[k] * CI
                + 4.55e-6 * AGE[k] * FI
                + 0.643 * INI_IRI
                - 2.5e-2 * AGE_IRI
                - 2.4e-6 * AGE[k] * RAIN
                - 1.09e-5 * AGE[k] * FT
            ) - 0.1,
            4
        )
        IRI.append(iri_k)

    # ================= Rutting =================
    rutting = []
    for k in range(len(AGE)):
        r_k = round(
            math.exp(
                a_rutting
                + 0.503 * math.log(AGE[k] + 0.1)
                + 3.37e-4 * CI
                + 1.22e-5 * RAIN
                + 3.48e-3 * FT
                + 2.98e-7 * FI * RAIN
                - 8.44e-2 * (math.log10(ESAL)/SN) * (math.log(AGE[k] + 0.1))
                - 1.42e-4 * (math.log(AGE[k] + 0.1)) * CI
                - 1.38e-5 * (math.log(AGE[k] + 0.1)) * FI
                + float(beta) * (math.log10(ESAL)/SN)
                + float(gama) * FI
            ) - 0.1,
            4
        )
        rutting.append(r_k)

    # ================= Fatigue =================
    fatigue = []
    # linear polynomial for first crack age
    p = [
        0.513 - 6.1e-2 * (math.log10(ESAL)/SN) - 1.15e-4 * CI - 1.44e-4 * FI,
        -math.log(0.7/(1-0.7)) + a1_fatigue + 0.917 * (math.log10(ESAL)/SN) - 1.12e-2 * FT
        + 1.06e-4 * FI - 2.21e-4 * CI - 1.01e-3 * RAIN
    ]
    roots = np.roots(p)
    first_crack_age = float(roots[0]) if roots.size > 0 else 0.0
    if first_crack_age < 0:
        first_crack_age = 0.0

    for k in range(len(AGE)):
        if AGE[k] < first_crack_age:
            f_k = 0.0
        else:
            f_k = np.round(
                math.exp(
                    a2_fatigue
                    + 7.63e-2 * (math.log10(ESAL)/SN)
                    + 0.737 * math.log(AGE[k] - first_crack_age + 0.1)
                    - 1.04e-3 * CI
                    - 4.12e-4 * FI
                    + 2.03e-5 * RAIN
                    - 1.12e-2 * FT
                    + 2.07e-4 * RAIN * math.log(AGE[k] - first_crack_age + 0.1)
                ) - 0.1,
                4
            )
        fatigue.append(float(f_k))

    # --- publish back to globals so downstream code sees the fresh series ---
    g["IRI"]     = np.asarray(IRI, dtype=float)
    g["rutting"] = np.asarray(rutting, dtype=float)
    g["fatigue"] = np.asarray(fatigue, dtype=float)

def compute_and_export():
    # Ensure output directory globals exist
    global out_dir, outdir
    if "out_dir" not in globals() or out_dir is None:
        sec_name = safe_folder_name(globals().get("Section_Name", "section"))

        try:
            base = self.output_root
        except Exception:
            base = Path.cwd() / "outputs"
        out_dir = base / sec_name
        out_dir.mkdir(parents=True, exist_ok=True)
    outdir = out_dir  # keep both names in sync

    rebuild_baseline_series_from_globals()

    # Distress-specific RSIs
    r1 = rsi_three_stage(IRI,     IRI_preservation_threshold,     IRI_rehabilitation_threshold,     IRI_reconstruction_threshold,     Design_life)
    r2 = rsi_three_stage(rutting, rutting_preservation_threshold, rutting_rehabilitation_threshold, rutting_reconstruction_threshold, Design_life)
    r3 = rsi_three_stage(fatigue, fatigue_preservation_threshold, fatigue_rehabilitation_threshold, fatigue_reconstruction_threshold, Design_life)

    # Combined (earliest crossing)
    global RSI_preservation, RSI_rehabilitation, RSI_reconstruction
    RSI_preservation   = min(r1[0], r2[0], r3[0])
    RSI_rehabilitation = min(r1[1], r2[1], r3[1])
    RSI_reconstruction = min(r1[2], r2[2], r3[2])

    print("RSI_preservation:", RSI_preservation)
    print("RSI_rehabilitation:", RSI_rehabilitation)
    print("RSI_reconstruction:", RSI_reconstruction)

    # ---- Run the exhaustive enumeration (generates one PDF per unique area) ----
    all_runs = enumerate_all_sequences()

    def _is_number(x):
        try:
            return math.isfinite(float(x))
        except Exception:
            return False
    
    # Keep only real, terminal results with a numeric area
    runs = [r for r in all_runs
            if isinstance(r, dict)
            and r.get('terminal') is True
            and _is_number(r.get('area'))]
    
    # If somehow nothing made it through, you can force-add a DN run (optional)
    if not runs:
        dn = _simulate_sequence([0], render_terminal=True)
        if isinstance(dn, dict) and dn.get('terminal') and _is_number(dn.get('area')):
            runs = [dn]
        else:
            raise RuntimeError("No valid terminal runs with numeric 'area' were produced for this section.")

    # Pick base case
    valid_runs = [r for r in runs if _safe_float(r.get('area')) is not None]
    dn_candidates = [r for r in valid_runs if tuple(r.get('seq', ())) == tuple()]
    if dn_candidates:
        base_run = dn_candidates[0]
    else:
        # Ensure DN exists (it usually does via the [0] branch in DFS, but just in case)
        dn_run = _simulate_sequence([0], render_terminal=True)
        runs.append(dn_run)
        base_run = dn_run
    
    # ---- Agency scaling by lane-kilometers (section length in km × number of lanes) ----
    len_km_raw   = globals().get("LEN", 1.0)     # Section length (km) from inputs
    lanes_raw    = globals().get("LANES", 1.0)   # Number of lanes from inputs
    LEN_KM   = float(len_km_raw or 1.0)
    LANES_NO = float(lanes_raw  or 1.0)
    LK_FACTOR = LEN_KM * LANES_NO  # lane-kilometers in this section

    def _scaled_npvs(run):
        cost_agency_per_lkm = float(run.get('cost_agency', 0.0))   # per lane-km
        cost_user_total     = float(run.get('cost_user',   0.0))   # total already
        cost_agency_total   = cost_agency_per_lkm * LK_FACTOR 
        cost_total_total    = cost_agency_total + cost_user_total
        return cost_agency_total, cost_user_total, cost_total_total
    
    base_agency, base_user, base_total = _scaled_npvs(base_run)
    
    # Build one-row-per-alternative table
    rows = []
    for r in runs:
        a = _safe_float(r.get('area'))
        area_val = round(float(r.get('area', 0.0)), 2)
        cost_agency, cost_user, cost_total = _scaled_npvs(r)
    
        # Benefits = Base − Alternative (positive = savings vs base)
        ben_ag = base_agency - cost_agency
        ben_us = base_user   - cost_user
        ben_tt = ben_ag + ben_us
    
        rows.append({
            "Seq": r.get("seq", ()),
            "Area": area_val,
            "NPV_Agency": round(ben_ag, 2),
            "NPV_User": round(ben_us, 2),
            "NPV_Total": round(ben_tt, 2),
            "Agency_Cost_Abs": round(cost_agency, 2)
        })

    # Sort if you like (by Area, or by NPV_Total, etc.)
    rows = sorted(rows, key=lambda d: (-d["Area"], -d["NPV_Total"]))
    df_results = pd.DataFrame(rows, columns=[
        "Seq","Area","NPV_Agency","NPV_User","NPV_Total","Agency_Cost_Abs"
    ])

    # Collect per-year costs tables (only for terminal runs that returned them)
    cost_tables = []
    for r in runs:
        if isinstance(r.get("df_costs"), pd.DataFrame):
            dfc = r["df_costs"].copy()
            dfc.insert(0, "Seq", "-".join(map(str, r.get('seq', ()))) or "∅")
            cost_tables.append(dfc)
    
    df_per_year = pd.concat(cost_tables, ignore_index=True) if cost_tables else pd.DataFrame(
        columns=["Area","Year","Agency($/yr)","User($/yr)","Total($/yr)","DF","Agency_PV","User_PV","Total_PV"]
    )
    
    # ========= MCDA: Normalize (exclude do-nothing from ranges) → Pareto (min Area, max NPV) → Knee =========
    # Ensure an Alt label exists (for plotting / tables)
    if "Alt" not in df_results.columns:
        df_results = df_results.copy()
        df_results.insert(0, "Alt", [f"Alt_{i+1}" for i in range(len(df_results))])
    
    # Build normalization dataset from project alternatives (no extra "Base" row here)
    norm_df = df_results.loc[:, ["Alt","Seq","Area","NPV_Total"]].copy()
    
    # ---- Identify special cases
    def _as_seq_tuple(val):
        if isinstance(val, (list, tuple)):
            return tuple(int(x) for x in val)
        s = str(val).strip()
        if s in ("", "None", "∅"): return tuple()
        if s.startswith("(") and s.endswith(")"): s = s[1:-1]
        parts = s.split("-") if ("-" in s and "," not in s) else [p.strip() for p in s.split(",") if p.strip()]
        try:    return tuple(int(p) for p in parts) if parts else tuple()
        except: return tuple()
    
    seqs = norm_df["Seq"].apply(_as_seq_tuple)
    norm_df["Is_DoNothing"] = seqs.apply(lambda t: len(t) == 0)
    
    # Ranges & selection: exclude ONLY do-nothing
    mask_norm = ~norm_df["Is_DoNothing"]
    mask_sel  = ~norm_df["Is_DoNothing"]

    # ---- Area: min–max on non-do-nothing; apply to all
    A_all = norm_df["Area"].astype(float).to_numpy()
    A_sub = A_all[mask_norm]
    if A_sub.size and np.isfinite(A_sub).any():
        Amin, Amax = float(np.nanmin(A_sub)), float(np.nanmax(A_sub))
        u_Area_all = (A_all - Amin) / (Amax - Amin) if Amax > Amin else np.full_like(A_all, 0.5, float)
    else:
        u_Area_all = np.full_like(A_all, 0.5, float)
    norm_df["Area_norm01"] = u_Area_all   # 0 best, 1 worst
    
    # ---- NPV: rules computed on non-do-nothing; applied to all
    N_all = norm_df["NPV_Total"].astype(float).to_numpy()
    N_sub = N_all[mask_norm & np.isfinite(N_all)]
    if N_sub.size == 0:
        u_NPV_all = np.full_like(N_all, np.nan, float)
    else:
        nmin, nmax = float(np.nanmin(N_sub)), float(np.nanmax(N_sub))
        if (nmax <= 0) or (nmin >= 0):
            den = nmax - nmin
            u_NPV_all = (N_all - nmin) / den if den > 0 else np.full_like(N_all, 0.5, float)
        else:
            # mixed: keep relative differences among positives; clamp ≤0 to 0
            pos_mask_sub = N_sub > 0
            Mpos = float(np.nanmax(N_sub[pos_mask_sub])) if np.any(pos_mask_sub) else 0.0
        
            u_NPV_all = np.zeros_like(N_all, dtype=float)   # default 0 for ≤0 (incl. base=0)
            if Mpos > 0:
                u_NPV_all[N_all > 0] = N_all[N_all > 0] / Mpos
            # else: all non-negatives are 0 -> remain 0
    norm_df["NPV_norm01"] = u_NPV_all
    
    # ---- Eligibility (exclude base from selection)
    eligible = mask_sel & np.isfinite(norm_df["Area_norm01"]) & np.isfinite(norm_df["NPV_norm01"])
    sel = norm_df.loc[eligible, ["Alt","Seq","Area","NPV_Total","Area_norm01","NPV_norm01"]].copy()
    
    # ---- Pareto (minimize Area_norm01, maximize NPV_norm01)
    def _pareto_minA_maxB(M: np.ndarray) -> np.ndarray:
        n = M.shape[0]; keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]: continue
            dom = (M[:,0] <= M[i,0]) & (M[:,1] >= M[i,1]) & ((M[:,0] < M[i,0]) | (M[:,1] > M[i,1]))
            if np.any(dom): keep[i] = False
        return keep
    
    if len(sel):
        M = sel[["Area_norm01","NPV_norm01"]].to_numpy(float)
        sel["Pareto"] = _pareto_minA_maxB(M)
        pareto_set = sel[sel["Pareto"]].copy()
    else:
        sel["Pareto"] = []
        pareto_set = sel.iloc[0:0].copy()
    
    # --- KNEE via max perpendicular distance to the chord (elbow) ---
    sel["Knee_Selected"] = False
    sel["Dist_to_Chord"] = np.nan
    
    if len(pareto_set) >= 2:
        # Sort front left→right on Area (smaller better) and top→down on NPV (larger better)
        P = pareto_set.sort_values(["Area_norm01","NPV_norm01"], ascending=[True, False]).copy()
    
        # Benefit space (both axes: larger = better)
        xb = 1.0 - P["Area_norm01"].to_numpy(float)
        yb = P["NPV_norm01"].to_numpy(float)
        pts = np.column_stack([xb, yb])
    
        a = pts[0]; b = pts[-1]; v = b - a
        denom = np.linalg.norm(v) + 1e-12
        w = pts - a
        dist = np.abs(v[0]*w[:,1] - v[1]*w[:,0]) / denom
    
        knee_idx = P.index[int(np.argmax(dist))]          # <-- set knee index
        sel.loc[knee_idx, "Knee_Selected"] = True
        sel.loc[P.index, "Dist_to_Chord"] = dist
    
    elif len(pareto_set) == 1:
        knee_idx = pareto_set.index[0]
        sel.loc[knee_idx, "Knee_Selected"] = True
    # else: no Pareto points → leave knee_idx as None
    
    # ---- Reflect flags on full table (for the normalization sheet)
    norm_df["Eligible_for_Selection"] = eligible
    norm_df["Pareto"] = False
    if len(pareto_set):
        norm_df.loc[pareto_set.index, "Pareto"] = True
    
    # ensure columns exist
    if "Knee_Selected" not in norm_df.columns:
        norm_df["Knee_Selected"] = False
    else:
        norm_df["Knee_Selected"] = False
    
    if "Dist_to_Chord" not in norm_df.columns:
        norm_df["Dist_to_Chord"] = np.nan
    
    if knee_idx is not None:
        norm_df.loc[knee_idx, "Knee_Selected"] = True

        if len(pareto_set) >= 2:
            norm_df.loc[P.index, "Dist_to_Chord"] = dist
            
    # ---- Plot (save next to Results.xlsx)
    plot_path = outdir / "pareto_knee_plot.png"
    plt.figure(figsize=(7,6))
    if len(sel):
        plt.scatter(sel["Area_norm01"], sel["NPV_norm01"], s=60, label="Eligible")
    if len(pareto_set):
        Pp = pareto_set.sort_values(["Area_norm01","NPV_norm01"], ascending=[True, False])
        plt.scatter(Pp["Area_norm01"], Pp["NPV_norm01"], s=70, label="Pareto")
        if len(Pp) >= 2:
            plt.plot(Pp["Area_norm01"], Pp["NPV_norm01"], ls="--", alpha=0.7)
    # Mark DN
    dn_pts = norm_df[norm_df["Is_DoNothing"]]
    plt.scatter([0],[1], marker="*", s=120, label="Ideal (0,1)")
    plt.xlabel("Area (min–max; 0 best)")
    plt.ylabel("NPV (rules; 1 best)")
    plt.title("Pareto set with knee")
    plt.xlim(-0.05, 1.05); plt.ylim(-0.05, 1.05); plt.grid(True, ls=":"); plt.legend()
    
    # ---- Annotate the knee safely
    if knee_idx is not None:
        knee = sel.loc[knee_idx]  # now safe
        xk = float(knee["Area_norm01"]); yk = float(knee["NPV_norm01"])
        seq_val = knee.get("Seq", "")
        if isinstance(seq_val, (list, tuple)):
            seq_str = "-".join(map(str, seq_val)) if len(seq_val) else "∅"
        else:
            seq_str = str(seq_val) if str(seq_val) not in ("", "None") else "∅"
    
        label = (
            f"Knee: {knee['Alt']}\n"
            f"Seq: {seq_str}\n"
            f"Area = {knee['Area']:.2f}\n"
            f"NPV  = {knee['NPV_Total']:,.2f}"
        )
        dx = -40 if xk > 0.6 else 40
        dy = -40 if yk > 0.6 else 40
        plt.annotate(
            label, xy=(xk, yk), xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top",
            arrowprops=dict(arrowstyle="->", lw=1),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.95)
        )
    
    plt.savefig(plot_path, dpi=200, bbox_inches="tight"); plt.close()

    outpath = out_dir / "Results.xlsx"
    with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
        # Existing sheets
        df_results.to_excel(writer, sheet_name="Summary", index=False)
        df_per_year.to_excel(writer, sheet_name="PerYearCosts", index=False)
    
        # normalization (incl. flags for base)
        norm_cols = [
            "Alt","Seq","Area","NPV_Total","Area_norm01","NPV_norm01",
            "Is_DoNothing","Eligible_for_Selection","Pareto","Knee_Selected","Dist_to_Chord"
        ]
        norm_df[norm_cols].to_excel(writer, sheet_name="MCDA_Normalization", index=False)
    
        # Pareto table (eligible only)
        sel_cols = ["Alt","Seq","Area","NPV_Total","Area_norm01","NPV_norm01","Pareto","Knee_Selected","Dist_to_Chord"]
        sel[sel_cols].sort_values(["Knee_Selected","Pareto","NPV_Total"], ascending=[False, False, False]) \
                     .to_excel(writer, sheet_name="MCDA_Pareto", index=False)
    
        # plot sheet
        workbook = writer.book
        ws_plot = workbook.add_worksheet("ParetoPlot")
        writer.sheets["ParetoPlot"] = ws_plot
        ws_plot.insert_image("B2", plot_path)
    
    print(f"Excel updated with MCDA sheets & plot → {outpath}")

# ---------- batch runner ----------
def run_batch(csv_path: str, output_root: Path):
    # No UI; reuse your CSV translation + fill helpers
    app = PavementEditor(output_root=output_root)

    import csv
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)  # assumes comma-delimited; adjust if needed
        rows = [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
                for row in rdr]

    total = len(rows)
    for i, raw in enumerate(rows, 1):
        row = app._translate_headers(raw)
        app._fill_from_dict(row)
        app.on_save(silent=True)     # pushes values to globals and creates section folder
        for k in ("IRI", "rutting", "fatigue"):
            globals().pop(k, None)
        compute_and_export()
        sec = row.get("Section_Name") or row.get("Section name") or f"Row {i}"
        print(f"[{i}/{total}] done: {sec}")

if __name__ == "__main__":
    if HEADLESS:
        run_batch(BATCH_CSV, OUTPUT_ROOT)
        sys.exit(0)
    else:
        # interactive UI
        def run_wizard_one_window():
            app = PavementEditor(output_root=OUTPUT_ROOT)
            state = app.run()
            return state
        run_wizard_one_window()

