#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graph_ui.py — ErrorGraph (CRITT M2A)
Comparer deux graphes (image ou tableau). Palette/Pipette.
Calibration par clic (origine bas-gauche, coin haut-droit) pour mapper pixels→données.
Extraction robuste (Lab+squelette), erreur %, export CSV/PNG, log, help, fermeture propre.
Auteur : Idir ARSLANE — © 2025
"""

import os, sys, json, random, string
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, simpledialog
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image, ImageTk
from skimage.morphology import skeletonize, remove_small_objects

# --- chemins ressources ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "ressources")
LOGO_PATH = os.path.join(RES_DIR, "Logo CRITT M2A.png")

# --- dossier des réglages ---
SETTINGS_DIR = os.path.join(BASE_DIR, "MySettings")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "last_settings.json")

def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] chargement settings: {e}")
    return {}

def save_settings(d: dict):
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

# ----- dépendances optionnelles (XLSX/MAT) -----
try:
    import pandas as pd
except Exception:
    pd = None
try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= Extraction / Comparaison =================

@dataclass
class HSVRange:
    low: Tuple[int,int,int]
    high: Tuple[int,int,int]
    low2: Optional[Tuple[int,int,int]] = None
    high2: Optional[Tuple[int,int,int]] = None

def hex_to_bgr(hex_color: str) -> Tuple[int,int,int]:
    h = hex_color.lstrip("#")
    r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
    return (b,g,r)

def bgr_to_hsv_range(bgr: Tuple[int,int,int], tol_h=8, tol_s=60, tol_v=60) -> HSVRange:
    hsv = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2HSV)[0,0]
    h,s,v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    lh, hh = max(0,h-tol_h), min(179,h+tol_h)
    ls, hs = max(0,s-tol_s), min(255,s+tol_s)
    lv, hv = max(0,v-tol_v), min(255,v+tol_v)
    if h < tol_h or h + tol_h > 179:
        return HSVRange((0,ls,lv),(min(hh,10),hs,hv),(max(170,179-tol_h),ls,lv),(179,hs,hv))
    return HSVRange((lh,ls,lv),(hh,hs,hv))

# ================= Extraction robuste =================

def _deltaE76_lab(lab_img, lab_ref):
    d = lab_img - lab_ref.reshape(1,1,3)
    return np.sqrt(np.sum(d*d, axis=2))

def detect_plot_roi(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 230), (180, 30, 255))
    nonwhite = cv2.bitwise_not(white)
    k = np.ones((7,7), np.uint8)
    nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, k, 2)
    cnts, _ = cv2.findContours(nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w = img_bgr.shape[:2]; return (0,0,w,h)
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(0.02*max(w,h))
    x = max(0, x-pad); y = max(0, y-pad)
    w = min(img_bgr.shape[1]-x, w+2*pad); h = min(img_bgr.shape[0]-y, h+2*pad)
    return (x,y,w,h)

def _color_mask_lab(img_bgr, hex_color, dE_thr=22, sat_min=25):
    bgr_ref = np.array(hex_to_bgr(hex_color), dtype=np.uint8).reshape(1,1,3)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2LAB).astype(np.float32)[0,0,:]
    dE = _deltaE76_lab(lab, lab_ref)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    mask = (dE <= dE_thr) & (S >= sat_min)
    return (mask.astype(np.uint8)*255)

def _keep_longest_component(mask, min_span_ratio=0.35):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1: return mask
    h,w = mask.shape
    best=-1; best_span=-1
    for i in range(1,n):
        x,y,ww,hh,area = stats[i]
        if area < 100: continue
        span = ww / max(1,w)
        if span > best_span: best_span=span; best=i
    if best < 0 or best_span < min_span_ratio: return mask
    out = np.zeros_like(mask); out[labels==best] = 255
    return out

def _bridge(mask, it=2, width=9):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (width,1))
    return cv2.dilate(mask, k, iterations=it)

def _skeleton(mask):
    sk = skeletonize((mask>0).astype(bool))
    sk = remove_small_objects(sk, 30, connectivity=2)
    return (sk.astype(np.uint8)*255)

def _trace_centerline(sk):
    h,w = sk.shape
    xs,ys=[],[]
    for u in range(w):
        col = np.where(sk[:,u]>0)[0]
        if col.size>0:
            xs.append(u); ys.append(int(np.median(col)))
    if len(xs)<5: return None
    xs=np.array(xs); ys=np.array(ys)
    x_full = np.arange(0,w)
    y_full = np.interp(x_full, xs, ys)
    return np.column_stack([x_full, y_full])

def robust_extract_curve(img_bgr, hex_color: str, limits=None, calib_pixels=None):
    """
    calib_pixels: ((x0,y0),(x1,y1)) en px image pour imposer le ROI.
    limits = (xmin,xmax,ymin,ymax) mappées sur ce ROI.
    """
    if calib_pixels is not None:
        (x0,y0),(x1,y1) = calib_pixels
        x = int(min(x0,x1)); x2 = int(max(x0,x1))
        y = int(min(y0,y1)); y2 = int(max(y0,y1))
        w = max(1, x2-x); h = max(1, y2-y)
    else:
        x,y,w,h = detect_plot_roi(img_bgr)

    roi = img_bgr[y:y+h, x:x+w]
    m0 = _color_mask_lab(roi, hex_color)
    m1 = _keep_longest_component(m0, min_span_ratio=0.35)
    m2 = _bridge(m1, it=2, width=max(7, w//120))
    sk = _skeleton(m2)
    path = _trace_centerline(sk)
    if path is None:
        raise RuntimeError("Courbe non trouvée. Ajuster couleur/pipette.")
    u = path[:,0]; v = path[:,1]
    xn = u / max(1.0, w-1)
    yn = 1.0 - v / max(1.0, h-1)
    if limits is None:
        return np.column_stack([xn, yn])
    xmin,xmax,ymin,ymax = limits
    xd = xmin + xn*(xmax-xmin)
    yd = ymin + yn*(ymax-ymin)
    return np.column_stack([xd, yd])

# ----- utilitaires comparaison -----

def resample_xy(xy, x_new):
    x=xy[:,0]; y=xy[:,1]; idx=np.argsort(x)
    return np.column_stack([x_new, np.interp(x_new, x[idx], y[idx])])

def compute_metrics(y_ref, y_mod):
    eps=1e-12; d=y_mod-y_ref
    rmse=float(np.sqrt(np.mean(d**2))); mae=float(np.mean(np.abs(d)))
    mape=float(np.mean(np.abs(d)/np.maximum(eps,np.abs(y_ref)))*100.0)
    ss_res=float(np.sum(d**2)); ss_tot=float(np.sum((y_ref-np.mean(y_ref))**2)+eps)
    r2=1.0-ss_res/ss_tot
    return {"RMSE":rmse,"MAE":mae,"MAPE_%":mape,"R2":r2}

# ===================== Thème MATRIX =====================

MATRIX_BG  = "#000000"
MATRIX_FG  = "#00FF41"   # vert matrix
MATRIX_ACC = "#00B000"

def apply_matrix_theme(root: tk.Tk):
    root.configure(bg=MATRIX_BG)
    style = ttk.Style(root)
    try: style.theme_use("clam")
    except tk.TclError: pass

    style.configure(".", background=MATRIX_BG, foreground=MATRIX_FG)
    style.configure("TLabel", background=MATRIX_BG, foreground=MATRIX_FG)
    style.configure("TFrame", background=MATRIX_BG)
    style.configure("TLabelframe", background=MATRIX_BG, foreground=MATRIX_FG, bordercolor=MATRIX_ACC)
    style.configure("TLabelframe.Label", background=MATRIX_BG, foreground=MATRIX_FG)
    style.configure("TButton", background=MATRIX_BG, foreground=MATRIX_FG, bordercolor=MATRIX_ACC,
                    focusthickness=3, focuscolor=MATRIX_ACC, padding=4)
    style.map("TButton", background=[("active", MATRIX_ACC)], foreground=[("active", MATRIX_BG)])
    style.configure("TEntry", fieldbackground="#101010", foreground=MATRIX_FG, insertcolor=MATRIX_FG, bordercolor=MATRIX_ACC)
    style.configure("TCombobox", fieldbackground="#101010", foreground=MATRIX_FG, arrowcolor=MATRIX_FG)
    style.configure("Treeview", background="#101010", foreground=MATRIX_FG, fieldbackground="#101010", bordercolor=MATRIX_ACC)
    style.configure("Vertical.TScrollbar", background=MATRIX_BG, troughcolor="#101010", arrowcolor=MATRIX_FG)
    root.option_add("*Foreground", MATRIX_FG)
    root.option_add("*Background", MATRIX_BG)
    root.option_add("*highlightColor", MATRIX_ACC)
    root.option_add("*activeForeground", MATRIX_BG)
    root.option_add("*activeBackground", MATRIX_ACC)

def style_axes(ax):
    ax.set_facecolor(MATRIX_BG)
    ax.tick_params(colors=MATRIX_FG, labelcolor=MATRIX_FG)
    for spine in ax.spines.values():
        spine.set_color(MATRIX_FG)
    ax.xaxis.label.set_color(MATRIX_FG)
    ax.yaxis.label.set_color(MATRIX_FG)
    ax.title.set_color(MATRIX_FG)
    ax.grid(True, color=MATRIX_FG, alpha=0.25)

# --- Matrix Help / Collapsible sections with decrypt effect -------------------
_MATRIX_GLYPHS = list("01" + string.ascii_uppercase + "░▒▓█/\\|<>[]{}#$%&@+=*")
MATRIX_FONT = ("Consolas", 10)

def _decrypt_animate(label, final_text: str, step_ms=16, jitter=2):
    state = {"i": 0, "final": final_text, "buf": list(final_text)}
    def tick():
        i = state["i"]; fin = state["final"]; buf = state["buf"]
        span = max(1, len(fin)//60); end = min(len(fin), i + span)
        for k in range(i, len(fin)):
            if fin[k] == "\n":
                buf[k] = "\n"
            elif k < end:
                buf[k] = fin[k]
            else:
                if random.randint(0, jitter) == 0:
                    buf[k] = random.choice(_MATRIX_GLYPHS)
        label.config(text="".join(buf))
        state["i"] = end
        if end < len(fin):
            label.after(step_ms, tick)
    tick()

class CollapsibleSection(tk.Frame):
    def __init__(self, master, title: str, body_text: str, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.configure(bg=MATRIX_BG, highlightthickness=1, highlightbackground=MATRIX_ACC)
        self._open = False; self._title = title; self._body_text = body_text
        self.header = tk.Frame(self, bg=MATRIX_BG); self.header.pack(fill="x")
        self.btn = tk.Button(self.header, text="+", width=2, command=self.toggle,
                             bg=MATRIX_BG, fg=MATRIX_FG, activebackground=MATRIX_BG,
                             activeforeground=MATRIX_FG, relief="flat", font=MATRIX_FONT, cursor="hand2")
        self.btn.pack(side="left", padx=(6, 2), pady=4)
        self.lbl = tk.Label(self.header, text=title, bg=MATRIX_BG, fg=MATRIX_FG,
                            font=("Consolas", 11, "bold"))
        self.lbl.pack(side="left", padx=2)
        self.body = tk.Frame(self, bg=MATRIX_BG)
        self.text = tk.Label(self.body, text="", justify="left", anchor="nw",
                             bg=MATRIX_BG, fg=MATRIX_FG, font=MATRIX_FONT, wraplength=760)
    def toggle(self):
        if self._open: self._collapse()
        else: self._expand()
    def _expand(self):
        self._open = True; self.btn.config(text="-")
        self.body.pack(fill="x", padx=6, pady=(0,8)); self.text.pack(fill="x")
        _decrypt_animate(self.text, self._body_text)
    def _collapse(self):
        self._open = False; self.btn.config(text="+")
        self.text.pack_forget(); self.body.pack_forget()

class MatrixHelpWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("ErrorGraph — Help")
        self.configure(bg=MATRIX_BG)
        self.geometry("900x720"); self.minsize(720, 560)
        outer = tk.Frame(self, bg=MATRIX_BG); outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=MATRIX_BG, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=MATRIX_BG)
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")
        header = tk.Label(
            frame,
            text=("=============================================\n"
                  "   ERRORGRAPH v1.0 — CRITT M2A © 2025\n"
                  "   Author : Idir ARSLANE\n"
                  "============================================="),
            bg=MATRIX_BG, fg=MATRIX_FG, font=("Consolas", 11), justify="center")
        header.pack(fill="x", pady=(10, 12))
        sections = [
            ("About",
             "Nom : ErrorGraph\nVersion : 1.0 Beta\nAuteur : Idir ARSLANE — CRITT M2A\nDate : 2025\nLicence : © CRITT M2A\n"),
            ("Features",
             "- Comparaison Image ↔ Tableau\n- Calibration par clic : origine (bas-gauche) + coin (haut-droit)\n"
             "- Couleurs : Palette (icône) + Pipette (icône)\n- Extraction robuste (Lab + squelette)\n"
             "- Fichiers : .png, .jpg, .csv, .xlsx, .mat\n- Exports : CSV, PNG\n- Thème Matrix complet (UI + plots)\n"),
            ("Usage rapide",
             "1) Charger Graph1 et Graph2 (image ou tableau).\n"
             "2) Si image : Calibrer → origine puis coin haut-droit → saisir Xmin/Xmax/Ymin/Ymax (défaut 0/1/0/1).\n"
             "3) Choisir la couleur (palette/pipette), extraire → affichage Matrix.\n"
             "4) ▶ Générer : erreurs (%) sur X commun.\n"
             "5) Exporter : overlay/error (PNG) et tableaux CSV.\n"),
            ("Options techniques",
             "- Limites par défaut : 0 / 1 / 0 / 1\n- Erreur % = |B - A| / max(eps, |A|) * 100\n"
             "- Recalage : linspace sur domaine commun\n- Tableaux : CSV (2 col.), XLSX (2 col.), MAT (x/y ou Nx2)\n"),
            ("Tips",
             "- Précision de la calibration = meilleure justesse.\n- Éviter les JPEG trop compressés.\n- Vérifier l’ordre des clics si inversion axe.\n"),
            ("Footer",
             "=============================================\nERRORGRAPH — The Matrix of Graphs\n>> Initializing... Done\n>> Ready for extraction.\n"
             "=============================================\n"),
        ]
        for title, body in sections:
            sec = CollapsibleSection(frame, title, body)
            sec.pack(fill="x", padx=10, pady=6)

def _open_first(root_frame):
    for child in root_frame.winfo_children():
        if isinstance(child, CollapsibleSection):
            child._expand()
            break

# --- Bandeau "pluie Matrix" couvrant tout le bandeau, pluie derrière texte ----
class MatrixBanner(tk.Canvas):
    """
    Canvas animé : pluie Matrix derrière un overlay (logo + titre + sous-titre).
    - Fond noir sur toute la largeur.
    - Pluie plus lente (step=10px, intervalle=100ms par défaut).
    - La pluie est dessinée avec le tag 'rain' et effacée sans toucher l'overlay.
    """
    def __init__(self, master, title_text, subtitle_text, logo_path=None,
                 height=84, step_px=10, interval_ms=100, **kwargs):
        super().__init__(master, height=height, bg="#000000",
                         highlightthickness=0, bd=0, **kwargs)
        self.pack_propagate(False)

        # pluie
        self._step = int(step_px)
        self._interval = int(interval_ms)
        self._chars = "01"
        self._cols = []
        self._init_cols()

        # overlay (titre + sous-titre sur fond noir)
        self._overlay_frame = tk.Frame(self, bg="#000000")
        self._overlay_id = self.create_window(10, height//2, window=self._overlay_frame,
                                              anchor="w")
        # contenu overlay
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).resize((56,56), Image.Resampling.LANCZOS)
                self._logo_img = ImageTk.PhotoImage(img)
                tk.Label(self._overlay_frame, image=self._logo_img, bg="#000000").pack(side="left", padx=(4,8))
            except Exception:
                pass
        block = tk.Frame(self._overlay_frame, bg="#000000")
        block.pack(side="left")

        tk.Label(block, text="ErrorGraph", font=("Segoe UI",18,"bold"),
                 fg=MATRIX_FG, bg="#000000").pack(anchor="w")
        tk.Label(block, text=title_text, font=("Segoe UI",10),
                 fg=MATRIX_FG, bg="#000000").pack(anchor="w")

        # events/anim
        self.bind("<Configure>", self._on_resize)
        self.after(self._interval, self._tick)

    def _on_resize(self, _evt=None):
        h = self.winfo_height()
        # recalc columns
        self._init_cols()
        # recenter overlay vertically, keep 10px left padding
        self.coords(self._overlay_id, 10, h//2)

    def _init_cols(self):
        w = max(1, self.winfo_width() or 1200)
        char_w = 12
        ncols = max(60, w // char_w)
        self._cols = [{
            "x": i * char_w + 6,
            "y": random.randint(-80, 0),
            "len": random.randint(5, 14)
        } for i in range(ncols)]

    def _tick(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if not w or not h:
            self.after(self._interval, self._tick)
            return

        # efface uniquement la pluie
        self.delete("rain")

        # dessine la pluie derrière (tag='rain')
        for c in self._cols:
            for k in range(c["len"]):
                y = c["y"] - k * 14
                if 0 <= y <= h:
                    ch = random.choice(self._chars)
                    color = "#00ff88" if k == 0 else "#105f40"
                    self.create_text(c["x"], y, text=ch, fill=color,
                                     font=("Consolas", 10, "bold"), tags=("rain",))
            c["y"] += self._step
            if c["y"] - c["len"] * 14 > h:
                c["y"] = random.randint(-80, 0)
                c["len"] = random.randint(5, 14)

        # s'assurer que l’overlay reste au-dessus
        self.tag_raise(self._overlay_id)
        self.after(self._interval, self._tick)
# -----------------------------------------------------------------------------


# ============================ UI ============================

class GraphColumn(ttk.LabelFrame):
    def __init__(self, master, title, on_change, settings_key: str):
        super().__init__(master, text=title, padding=8)
        self.on_change = on_change
        self.settings_key = settings_key

        # état
        self.mode = tk.StringVar(value="Image")
        self.color_hex = tk.StringVar(value="#00FF41")  # vert matrix par défaut
        self.hsv = bgr_to_hsv_range(hex_to_bgr(self.color_hex.get()))

        # limites (éditables)
        self.xmin_var = tk.DoubleVar(value=0.0)
        self.xmax_var = tk.DoubleVar(value=1.0)
        self.ymin_var = tk.DoubleVar(value=0.0)
        self.ymax_var = tk.DoubleVar(value=1.0)

        # Calibration par clic
        self._pipette_armed=False
        self._calib_armed = False
        self._calib_stage = 0  # 0=origine, 1=coin haut-droit
        self._img_disp_scale = (1.0, 1.0)
        self.calib_px0 = None
        self.calib_px1 = None

        # Données
        self.raw=None
        self.data=None

        self._build()

    @property
    def calib_data(self):
        return (float(self.xmin_var.get()), float(self.xmax_var.get()),
                float(self.ymin_var.get()), float(self.ymax_var.get()))
    @calib_data.setter
    def calib_data(self, tup):
        self.xmin_var.set(float(tup[0])); self.xmax_var.set(float(tup[1]))
        self.ymin_var.set(float(tup[2])); self.ymax_var.set(float(tup[3]))

    def _build(self):
        # Ligne 1 : Source + Charger  |  Couleur (à droite)
        row1 = ttk.Frame(self); row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Source").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.mode, values=["Image","Tableau"], width=10)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Charger", command=self.load).pack(side=tk.LEFT, padx=(6, 10))

        # Bloc couleur à droite
        color_blk = ttk.Frame(row1); color_blk.pack(side=tk.LEFT, padx=0)
        ttk.Label(color_blk, text="Couleur").pack(side=tk.LEFT, padx=(0,4))
        self.color_preview = tk.Canvas(color_blk, width=22, height=22,
                                       highlightthickness=1, highlightbackground=MATRIX_ACC, bg=MATRIX_BG)
        self.color_preview.pack(side=tk.LEFT)
        self._update_color_preview()
        ttk.Button(color_blk, text="🧪", width=3, command=self.arm_pipette).pack(side=tk.LEFT, padx=2)
        ttk.Button(color_blk, text="🎨", width=3, command=self.choose_custom_color).pack(side=tk.LEFT, padx=2)

        # Ligne 2 : 📐 Calibrer  + Xmin / Xmax
        row2 = ttk.Frame(self); row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="📐", width=3, command=self.arm_calibration).pack(side=tk.LEFT, padx=(0,6))
        ttk.Label(row2, text="Xmin").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.xmin_var, width=8).pack(side=tk.LEFT, padx=(2,6))
        ttk.Label(row2, text="Xmax").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.xmax_var, width=8).pack(side=tk.LEFT, padx=(2,0))

        # Ligne 3 : 🔄 Reset  + Ymin / Ymax
        row3 = ttk.Frame(self); row3.pack(fill=tk.X, pady=2)
        ttk.Button(row3, text="🔄", width=3, command=self.reset_calibration).pack(side=tk.LEFT, padx=(0,6))
        ttk.Label(row3, text="Ymin").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.ymin_var, width=8).pack(side=tk.LEFT, padx=(2,6))
        ttk.Label(row3, text="Ymax").pack(side=tk.LEFT)
        ttk.Entry(row3, textvariable=self.ymax_var, width=8).pack(side=tk.LEFT, padx=(2,0))

        # Aperçu image/table
        self.preview_frame=ttk.LabelFrame(self,text="Aperçu (clics: origine puis coin haut-droit)",padding=6)
        self.preview_frame.pack(fill=tk.BOTH,expand=True, pady=(4,0))
        self.preview_img_label=tk.Label(self.preview_frame, bg=MATRIX_BG, fg=MATRIX_FG)
        self.preview_img_label.pack_forget()
        self.preview_img_label.bind("<Button-1>", self._on_left_click_image)
        self.preview_img_label.bind("<Button-3>", self._cancel_tools)
        self.preview_img_label.bind("<Escape>",   self._cancel_tools)

        # Table 7 lignes + ascenseur
        self.preview_table_holder = ttk.Frame(self.preview_frame)
        self.preview_table = ttk.Treeview(
            self.preview_table_holder,
            columns=("x","y"),
            show="headings",
            height=7
        )
        self.preview_table.heading("x", text="x")
        self.preview_table.heading("y", text="y")
        for c in ("x","y"):
            self.preview_table.column(c, width=100, anchor="center")
        self.preview_vsb = ttk.Scrollbar(self.preview_table_holder, orient="vertical", command=self.preview_table.yview)
        self.preview_table.configure(yscrollcommand=self.preview_vsb.set)
        self.preview_table.grid(row=0, column=0, sticky="nsew")
        self.preview_vsb.grid(row=0, column=1, sticky="ns")
        self.preview_table_holder.columnconfigure(0, weight=1)
        self.preview_table_holder.rowconfigure(0, weight=1)
        self.preview_table_holder.pack_forget()

        # Figure
        self.fig,self.ax=plt.subplots(figsize=(3.6,2.6))
        self.fig.patch.set_facecolor(MATRIX_BG)
        style_axes(self.ax)
        self.canvas=FigureCanvasTkAgg(self.fig,master=self)
        self.canvas.get_tk_widget().configure(bg=MATRIX_BG, highlightbackground=MATRIX_ACC)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True,pady=6)

    # ---------- Couleur ----------
    def _update_color_preview(self):
        self.color_preview.delete("all")
        self.color_preview.create_rectangle(0,0,22,22,fill=self.color_hex.get(),outline="")

    def choose_custom_color(self):
        col = colorchooser.askcolor(title="Choisir une couleur")[1]
        if not col: return
        self._set_color(col)

    def _set_color(self, hexcol, win=None):
        self.color_hex.set(hexcol)
        self.hsv = bgr_to_hsv_range(hex_to_bgr(hexcol))
        self._update_color_preview()
        if win: win.destroy()
        self._refresh_filtered()
        self.on_change("palette")

    # ---------- Pipette ----------
    def arm_pipette(self):
        if self.raw is None or self.raw[0]!="IMG":
            messagebox.showinfo("Pipette","Charge d'abord une image."); return
        self._pipette_armed=True
        self._calib_armed=False
        self.preview_img_label.configure(cursor="crosshair")
        self.preview_img_label.focus_set()
        self.on_change("pipette_arm")

    def _cancel_tools(self, event=None):
        self._pipette_armed=False
        self._calib_armed=False
        self._calib_stage=0
        self.preview_img_label.configure(cursor="")
        self.on_change("tools_cancel")

    # ---------- Calibration ----------
    def arm_calibration(self):
        if self.raw is None or self.raw[0]!="IMG":
            messagebox.showinfo("Calibration","Charge d'abord une image."); return
        self._calib_armed=True
        self._pipette_armed=False
        self._calib_stage=0
        self.calib_px0=None; self.calib_px1=None
        self.preview_img_label.configure(cursor="tcross")
        self._info("Clique l’origine (bas-gauche), puis le coin haut-droit dans la zone de tracé.")

    def reset_calibration(self):
        self.calib_px0=None; self.calib_px1=None; self.calib_data=(0.0,1.0,0.0,1.0)
        self._info("Calibration réinitialisée (limites 0,1,0,1).")
        self._refresh_filtered()

    def _ask_data_limits(self):
        xmin = simpledialog.askfloat("Calibration données","Valeur Xmin (origine):", parent=self, initialvalue=self.xmin_var.get())
        if xmin is None: return None
        xmax = simpledialog.askfloat("Calibration données","Valeur Xmax (coin haut-droit):", parent=self, initialvalue=self.xmax_var.get())
        if xmax is None: return None
        ymin = simpledialog.askfloat("Calibration données","Valeur Ymin (origine):", parent=self, initialvalue=self.ymin_var.get())
        if ymin is None: return None
        ymax = simpledialog.askfloat("Calibration données","Valeur Ymax (coin haut-droit):", parent=self, initialvalue=self.ymax_var.get())
        if ymax is None: return None
        return (xmin,xmax,ymin,ymax)

    # ---------- Interactions image ----------
    def _on_left_click_image(self, event):
        if self.raw is None or self.raw[0]!="IMG":
            return
        scale_x, scale_y = self._img_disp_scale
        img_bgr = self.raw[3]
        ox = int(event.x * scale_x); oy = int(event.y * scale_y)
        ox = np.clip(ox, 0, img_bgr.shape[1]-1); oy = np.clip(oy, 0, img_bgr.shape[0]-1)

        if self._pipette_armed:
            b,g,r = map(int, img_bgr[oy, ox])
            hexcol = f"#{r:02X}{g:02X}{b:02X}"
            self._set_color(hexcol)
            self._cancel_tools()
            return

        if self._calib_armed:
            if self._calib_stage == 0:
                self.calib_px0 = (ox, oy)
                self._calib_stage = 1
                self._info(f"Origine enregistrée (px): {self.calib_px0}. Clique le coin haut-droit.")
            else:
                self.calib_px1 = (ox, oy)
                self._calib_stage = 0
                self._calib_armed = False
                self.preview_img_label.configure(cursor="")
                self._info(f"Coin haut-droit (px): {self.calib_px1}. Saisie des valeurs données…")
                lim = self._ask_data_limits()
                if lim is not None:
                    self.calib_data = lim
                    self._info(f"Calibration données: xmin={lim[0]}, xmax={lim[1]}, ymin={lim[2]}, ymax={lim[3]}")
                else:
                    self._info("Calibration données annulée. Limites actuelles conservées.")
                self._refresh_filtered()
            return

    # ---------- Chargement ----------
    def _load_table_from_path(self, path: str) -> Optional[np.ndarray]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            try:
                arr = np.loadtxt(path, delimiter=",", skiprows=1)
            except Exception:
                arr = np.loadtxt(path, delimiter=",")
            if arr.ndim==1 and arr.size>=2:
                arr = arr.reshape(-1,2)
            return arr[:,0:2]
        elif ext in (".xlsx", ".xls"):
            if pd is None:
                messagebox.showerror("Excel",
                    "pandas est requis pour lire .xlsx/.xls.\nInstallez : pip install pandas openpyxl")
                return None
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception:
                df = pd.read_excel(path)
            if df.shape[1] < 2:
                messagebox.showerror("Excel","Le fichier doit contenir au moins 2 colonnes (x,y).")
                return None
            return df.iloc[:,0:2].to_numpy(dtype=float)
        elif ext == ".mat":
            if loadmat is None:
                messagebox.showerror("MAT","scipy est requis pour lire .mat.\nInstallez : pip install scipy")
                return None
            try:
                md = loadmat(path)
            except Exception as e:
                messagebox.showerror("MAT", f"Impossible de lire {os.path.basename(path)}\n{e}")
                return None
            cand = None
            for k,v in md.items():
                if k.startswith("__"): continue
                a = np.asarray(v)
                if a.ndim==2 and a.shape[1]>=2:
                    cand = a[:,0:2]; break
            if cand is None:
                messagebox.showerror("MAT","Aucune variable (N×2) trouvée dans le .mat")
                return None
            return cand.astype(float)
        else:
            messagebox.showerror("Format","Formats supportés: .csv, .xlsx/.xls, .mat")
            return None

    def load(self):
        if self.mode.get()=="Image":
            path=filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
            if not path: return
            img_bgr=cv2.imread(path,cv2.IMREAD_COLOR)
            if img_bgr is None: messagebox.showerror("Erreur","Image illisible."); return
            h,w=img_bgr.shape[:2]; maxw,maxh=460,260
            sc=min(maxw/w,maxh/h); nw,nh=int(w*sc),int(h*sc)
            img_rgb=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
            pil=Image.fromarray(img_rgb).resize((nw,nh),Image.Resampling.LANCZOS)
            self._img_disp_scale=(w/nw, h/nh)
            tkimg=ImageTk.PhotoImage(pil)
            self.preview_img_label.configure(image=tkimg); self.preview_img_label.image=tkimg
            self.preview_img_label.pack(fill=tk.NONE,expand=False)
            self.preview_table_holder.pack_forget()
            self.raw=("IMG",path,pil,img_bgr)
            self._info(f"Image chargée: {os.path.basename(path)}  taille={w}x{h}")
            self._refresh_filtered()
        else:
            path=filedialog.askopenfilename(filetypes=[("Tableaux","*.csv;*.xlsx;*.xls;*.mat")])
            if not path: return
            arr = self._load_table_from_path(path)
            if arr is None: return
            self.raw=("CSV",arr[:,0:2])
            for r in self.preview_table.get_children(): self.preview_table.delete(r)
            for i in range(arr.shape[0]):
                self.preview_table.insert("", "end", values=(f"{arr[i,0]:.6g}", f"{arr[i,1]:.6g}"))
            self.preview_table_holder.pack(fill=tk.X,expand=False)
            self.preview_img_label.pack_forget()
            self._info(f"Table chargée: {os.path.basename(path)}  points={arr.shape[0]}")
            self._refresh_filtered()
        self.on_change("load")

    # ---------- Rendu ----------
    def _refresh_filtered(self):
        self.ax.clear()
        style_axes(self.ax)
        if self.raw is None:
            self.ax.set_title("Aucun signal"); self.canvas.draw_idle(); return

        if self.raw[0]=="CSV":
            self.data=self.raw[1].copy()
            self.ax.plot(self.data[:,0],self.data[:,1], color=MATRIX_FG, linewidth=1.5)
            self.ax.set_title("Tableau chargé")
        else:
            img_bgr=self.raw[3]
            try:
                calib_pix = None
                limits = self.calib_data
                if self.calib_px0 is not None and self.calib_px1 is not None:
                    calib_pix = (self.calib_px0, self.calib_px1)
                self.data = robust_extract_curve(img_bgr, self.color_hex.get(), limits, calib_pix)
                self.ax.plot(self.data[:,0],self.data[:,1], color=MATRIX_FG, linewidth=1.5)
                t = "Courbe extraite (calibrée)"
                self.ax.set_title(t)
            except Exception as e:
                self.data=None; self.ax.set_title(f"Extraction échouée: {e}")
        self.canvas.draw_idle()

    # ---------- Log util ----------
    def _info(self, msg):
        try:
            self.master.master._log(msg)
        except Exception:
            pass

    # ---------- settings ----------
    def get_settings(self) -> dict:
        return {
            "mode": self.mode.get(),
            "color": self.color_hex.get(),
            "limits": list(self.calib_data),
        }

    def set_settings(self, d: dict):
        if not d: return
        if "mode" in d: self.mode.set(d["mode"])
        if "color" in d:
            self.color_hex.set(d["color"]); self._update_color_preview()
        if "limits" in d and isinstance(d["limits"], (list, tuple)) and len(d["limits"])==4:
            self.calib_data = d["limits"]
        self._refresh_filtered()

# ============================ App ============================

class GraphUI(tk.Tk):
    def __init__(self):
        super().__init__()
        apply_matrix_theme(self)
        self.title("GRAPH — 01 0011 0101 • ErrorGraph (CRITT M2A)")
        self.geometry("1360x900")
        self.npoints=tk.IntVar(value=400)
        self.settings = load_settings()

        self._build_header(); self._build_body(); self._build_footer()

        # Charger derniers réglages
        self._apply_loaded_settings()

        self.compare_table=None; self.error_table=None
        self.A_data=None; self.B_data=None
        self._log("Prêt.")

    def _build_header(self):
        # Bandeau animé couvrant tout le bandeau, pluie derrière le texte
        MatrixBanner(
            self,
            title_text="Charge deux graphes (image/tableau), calibre par clic, extrait, superpose et calcule l’erreur.",
            subtitle_text="",  # (non utilisé, mais laissé pour extension)
            logo_path=LOGO_PATH,
            height=92,
            step_px=10,       # pas vertical plus petit => plus lent
            interval_ms=100   # intervalle augmenté => plus lent
        ).pack(side=tk.TOP, fill=tk.X)

    def _build_body(self):
        body=ttk.Frame(self); body.pack(fill=tk.BOTH,expand=True,padx=8,pady=6)
        body.columnconfigure(0, weight=1, uniform="cols")
        body.columnconfigure(1, weight=1, uniform="cols")
        body.columnconfigure(2, weight=0, minsize=72)
        body.columnconfigure(3, weight=1, uniform="cols")
        body.rowconfigure(0, weight=1)

        self.colA=GraphColumn(body,"Graph 1 (Biblio)",self._child_changed, settings_key="A")
        self.colA.grid(row=0,column=0,sticky="nsew",padx=6)
        self.colB=GraphColumn(body,"Graph 2 (Modèle)",self._child_changed, settings_key="B")
        self.colB.grid(row=0,column=1,sticky="nsew",padx=6)

        ttk.Button(body,text="▶ Générer",command=self.generate).grid(row=0,column=2,sticky="ns",padx=6)

        col3=ttk.LabelFrame(body,text="Erreur (%)",padding=8); col3.grid(row=0,column=3,sticky="nsew",padx=6)
        self.figErr,self.axErr=plt.subplots(figsize=(4.8,3.2))
        self.figErr.patch.set_facecolor(MATRIX_BG)
        style_axes(self.axErr)
        self.canvasErr=FigureCanvasTkAgg(self.figErr,master=col3)
        self.canvasErr.get_tk_widget().configure(bg=MATRIX_BG, highlightbackground=MATRIX_ACC)
        self.canvasErr.get_tk_widget().pack(fill=tk.BOTH,expand=True,pady=6)
        b=ttk.Frame(col3); b.pack(fill=tk.X)
        ttk.Button(b,text="Exporter image",command=self.export_images).pack(side=tk.LEFT,padx=4)
        ttk.Button(b,text="Exporter tableau",command=self.export_tables).pack(side=tk.LEFT,padx=4)

        rowg=ttk.Frame(body); rowg.grid(row=1,column=0,columnspan=4,sticky="w",padx=6,pady=(6,0))
        ttk.Label(rowg,text="N points").pack(side=tk.LEFT)
        ttk.Entry(rowg,textvariable=self.npoints,width=8).pack(side=tk.LEFT,padx=6)

    def _build_footer(self):
        foot=ttk.Frame(self); foot.pack(side=tk.BOTTOM,fill=tk.X,padx=8,pady=6)
        self.log=tk.Text(foot,height=8, bg="#101010", fg=MATRIX_FG, insertbackground=MATRIX_FG, highlightbackground=MATRIX_ACC)
        self.log.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
        r=ttk.Frame(foot); r.pack(side=tk.RIGHT,fill=tk.Y,padx=4)
        ttk.Button(r,text="Sauvegarder",command=self.save_current_settings).pack(fill=tk.X,pady=2)
        ttk.Button(r,text="Help",command=self.show_help).pack(fill=tk.X,pady=2)
        ttk.Button(r,text="Quitter",command=self.on_quit).pack(fill=tk.X,pady=2)

    def _apply_loaded_settings(self):
        s = self.settings if isinstance(self.settings, dict) else {}
        self.npoints.set(int(s.get("npoints", 400)))
        self.colA.set_settings(s.get("A", {}))
        self.colB.set_settings(s.get("B", {}))

    def save_current_settings(self):
        data = {
            "npoints": int(self.npoints.get()),
            "A": self.colA.get_settings(),
            "B": self.colB.get_settings(),
        }
        try:
            save_settings(data)
            self._log("Paramètres sauvegardés dans MySettings/last_settings.json")
        except Exception as e:
            messagebox.showerror("Sauvegarde", f"Impossible d’écrire les paramètres.\n{e}")

    def _log(self,msg): self.log.insert(tk.END,msg+"\n"); self.log.see(tk.END)
    def _child_changed(self,src): self._log(f"MAJ paramètres ({src}).")

    def generate(self):
        if self.colA.data is None or self.colB.data is None:
            messagebox.showwarning("Entrées","Charge/extrais Graph 1 et Graph 2."); return
        A,B=self.colA.data, self.colB.data
        self.A_data,self.B_data=A,B
        xmin=max(np.min(A[:,0]),np.min(B[:,0])); xmax=min(np.max(A[:,0]),np.max(B[:,0]))
        if xmax<=xmin: messagebox.showerror("Domaines","Pas de chevauchement en X."); return
        x_common=np.linspace(xmin,xmax,max(50,self.npoints.get()))
        A_rs=resample_xy(A,x_common); B_rs=resample_xy(B,x_common)
        m=compute_metrics(A_rs[:,1],B_rs[:,1])
        err_abs=np.abs(B_rs[:,1]-A_rs[:,1])
        err_pct=err_abs/np.maximum(1e-12,np.abs(A_rs[:,1]))*100.0
        self.compare_table=np.column_stack([A_rs[:,0],A_rs[:,1],B_rs[:,1],err_abs,err_pct])
        self.error_table=np.column_stack([A_rs[:,0],err_pct])
        self.axErr.clear()
        style_axes(self.axErr)
        self.axErr.plot(A_rs[:,0],err_pct,linewidth=1.5, color=MATRIX_FG)
        self.axErr.set_xlabel("x"); self.axErr.set_ylabel("Erreur [%]")
        self.axErr.set_title(f"Erreur relative (%) — MAPE={np.mean(err_pct):.2f}%")
        self.figErr.tight_layout(); self.canvasErr.draw_idle()
        self._log(f"Comparaison OK. RMSE={m['RMSE']:.3g}, MAE={m['MAE']:.3g}, MAPE={m['MAPE_%']:.2f}%, R²={m['R2']:.4f}")

    def export_images(self):
        if self.compare_table is None: messagebox.showinfo("Info","Génère d’abord les résultats."); return
        out=filedialog.askdirectory(title="Dossier d’export images")
        if not out: return
        fig,ax=plt.subplots(figsize=(7,6))
        fig.patch.set_facecolor(MATRIX_BG); style_axes(ax)
        ax.plot(self.A_data[:,0],self.A_data[:,1],label="Biblio",linewidth=2, color=MATRIX_FG)
        ax.plot(self.B_data[:,0],self.B_data[:,1],label="Modèle",linewidth=2, color="#80FF9F")
        ax.legend(facecolor=MATRIX_BG, edgecolor=MATRIX_FG)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("Superposition Biblio vs Modèle"); fig.tight_layout()
        fig.savefig(os.path.join(out,"overlay.png"),dpi=160, facecolor=MATRIX_BG)
        plt.close(fig)
        self.figErr.savefig(os.path.join(out,"error_percent.png"),dpi=160, facecolor=MATRIX_BG)
        self._log(f"Images exportées -> {out}")

    def export_tables(self):
        if self.compare_table is None: messagebox.showinfo("Info","Génère d’abord les résultats."); return
        out=filedialog.askdirectory(title="Dossier d’export tableaux")
        if not out: return
        np.savetxt(os.path.join(out,"series_A.csv"),self.A_data,delimiter=",",header="x,y",comments="")
        np.savetxt(os.path.join(out,"series_B.csv"),self.B_data,delimiter=",",header="x,y",comments="")
        np.savetxt(os.path.join(out,"compare_resampled.csv"),self.compare_table,delimiter=",",
                   header="x,y_biblio(A),y_modele(B),abs_error,pct_error",comments="")
        np.savetxt(os.path.join(out,"error_percent.csv"),self.error_table,delimiter=",",
                   header="x,error_percent",comments="")
        self._log(f"Tableaux exportés -> {out}")

    def show_help(self):
        MatrixHelpWindow(self)
        _open_first(self)  # ouvre la 1re section à l’ouverture

    def on_quit(self):
        try:
            self.save_current_settings()
        except Exception:
            pass
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = GraphUI()
    app.protocol("WM_DELETE_WINDOW", app.on_quit)
    app.mainloop()
