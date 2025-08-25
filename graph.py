#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparaison auto de deux graphes (Biblio vs Modèle) sans clic.
- Détection ROI du tracé.
- Extraction par couleur (HSV) + nettoyage morphologique.
- Conversion pixels -> données :
    * soit normalisée [0..1] si pas de limites
    * soit physique si --limits-a / --limits-b fournis
- Rééchantillonnage sur grille commune, métriques (RMSE, MAE, MAPE, R²).
- Exports CSV + overlay + courbe erreur %.

Exemples :
# Mode 100% auto (coordonnées normalisées)
python graph.py --img-a Biblio.png --img-b Modele.png --color-a blue --color-b blue --outdir out

# Avec limites physiques (si connues)
python graph.py --img-a Biblio.png --img-b Modele.png \
  --limits-a 0 9 0 25000 --limits-b 0 7 0 37000 \
  --color-a blue --color-b blue --outdir out
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt

@dataclass
class HSVRange:
    low: Tuple[int,int,int]
    high: Tuple[int,int,int]
    low2: Optional[Tuple[int,int,int]] = None
    high2: Optional[Tuple[int,int,int]] = None

PRESETS = {
    "blue":  HSVRange((90, 50, 50),  (135, 255, 255)),
    "red":   HSVRange((0, 70, 50),   (10, 255, 255), (170, 70, 50), (180, 255, 255)),
    "green": HSVRange((40, 40, 40),  (85, 255, 255)),
    "black": HSVRange((0, 0, 0),     (180, 255, 50)),
}

def mask_hsv(img_bgr, hsv_range: HSVRange):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(hsv_range.low), np.array(hsv_range.high))
    if hsv_range.low2 is not None:
        m2 = cv2.inRange(hsv, np.array(hsv_range.low2), np.array(hsv_range.high2))
        mask = cv2.bitwise_or(m1, m2)
    else:
        mask = m1
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 1)
    return mask

def detect_plot_roi(img_bgr, curve_mask):
    """
    Détecte une ROI de tracé robuste :
    - Combine bords Canny + masque courbe dilaté.
    - Prend la bbox du plus grand contour plausible.
    """
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dil_curve = cv2.dilate(curve_mask, kernel, iterations=2)
    combo = cv2.bitwise_or(edges, dil_curve)
    contours, _ = cv2.findContours(combo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # fallback: pleine image
        h, w = img_bgr.shape[:2]
        return (0, 0, w, h)
    # plus grand contour
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # marge pour axes
    pad = int(0.04*max(w,h))
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(img_bgr.shape[1]-x, w + 2*pad)
    h = min(img_bgr.shape[0]-y, h + 2*pad)
    return (x,y,w,h)

def extract_curve_points(img_bgr, hsv_range: HSVRange):
    mask = mask_hsv(img_bgr, hsv_range)
    x,y,w,h = detect_plot_roi(img_bgr, mask)
    roi = img_bgr[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]

    # extraction colonne par colonne dans la ROI
    us, vs = [], []
    for u in range(w):
        col = mask_roi[:, u]
        if np.count_nonzero(col) == 0:
            continue
        v_candidates = np.where(col > 0)[0]
        v = int(np.median(v_candidates))
        us.append(u); vs.append(v)
    if len(us) < 5:
        raise RuntimeError("Courbe non détectée. Ajuster couleurs.")
    pts_roi = np.column_stack([np.array(us), np.array(vs)])  # (u,v) dans ROI
    # repasse en coords image globale
    pts_img = pts_roi.copy()
    pts_img[:,0] += x
    pts_img[:,1] += y
    return pts_img, (x,y,w,h)

def map_pixels_to_data(pts_img, roi, limits=None):
    """
    Si limits=None -> normalisation [0..1] via ROI.
    Sinon -> mapping physique avec axes linéaires supposés alignés aux bords ROI.
    """
    x0,y0,w,h = roi
    u = pts_img[:,0]; v = pts_img[:,1]
    # coords locales à la ROI
    u_loc = (u - x0).astype(np.float64)
    v_loc = (v - y0).astype(np.float64)
    # normalisation [0..1]
    x_norm = u_loc / max(1.0, w-1)
    y_norm = 1.0 - (v_loc / max(1.0, h-1))  # y image vers haut

    if limits is None:
        return np.column_stack([x_norm, y_norm])
    xmin,xmax,ymin,ymax = limits
    x_data = xmin + x_norm * (xmax - xmin)
    y_data = ymin + y_norm * (ymax - ymin)
    return np.column_stack([x_data, y_data])

def resample_xy(xy, x_new):
    x = xy[:,0]; y = xy[:,1]
    idx = np.argsort(x)
    return np.column_stack([x_new, np.interp(x_new, x[idx], y[idx])])

def compute_metrics(y_ref, y_mod):
    eps = 1e-12
    diff = y_mod - y_ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae  = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff)/np.maximum(eps, np.abs(y_ref))) * 100.0)
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref))**2) + eps)
    r2   = 1.0 - ss_res/ss_tot
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}

def save_csv(path, arr, header):
    np.savetxt(path, arr, delimiter=",", header=header, comments="")

def plot_overlay(out_png, A_raw, B_raw, A_rs, B_rs, title_xy):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(A_raw[:,0], A_raw[:,1], label="Biblio (brut)", linewidth=2)
    ax.plot(B_raw[:,0], B_raw[:,1], label="Modèle (brut)", linewidth=2)
    ax.plot(A_rs[:,0], A_rs[:,1], "--", alpha=0.6, label="Biblio — rééchantillonné")
    ax.plot(B_rs[:,0], B_rs[:,1], "--", alpha=0.6, label="Modèle — rééchantillonné")
    ax.set_xlabel(title_xy[0]); ax.set_ylabel(title_xy[1])
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def plot_errpct(out_png, x, err_pct, xlab):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, err_pct, linewidth=1.5)
    ax.set_xlabel(xlab); ax.set_ylabel("Erreur [%]")
    ax.set_title("Erreur relative (%) Modèle vs Biblio"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    imgA = cv2.imread(args.img_a, cv2.IMREAD_COLOR)
    imgB = cv2.imread(args.img_b, cv2.IMREAD_COLOR)
    if imgA is None or imgB is None:
        raise FileNotFoundError("Image introuvable.")

    hsvA = PRESETS.get(args.color_a.lower(), PRESETS["blue"])
    hsvB = PRESETS.get(args.color_b.lower(), PRESETS["blue"])

    # Extraction auto
    ptsA_img, roiA = extract_curve_points(imgA, hsvA)
    ptsB_img, roiB = extract_curve_points(imgB, hsvB)

    # Mapping
    limitsA = tuple(args.limits_a) if args.limits_a else None
    limitsB = tuple(args.limits_b) if args.limits_b else None

    A_data = map_pixels_to_data(ptsA_img, roiA, limitsA)
    B_data = map_pixels_to_data(ptsB_img, roiB, limitsB)

    # Sauvegarde bruts
    save_csv(os.path.join(args.outdir, "series_A.csv"), A_data, "x,y")
    save_csv(os.path.join(args.outdir, "series_B.csv"), B_data, "x,y")

    # Grille commune
    xmin = max(np.min(A_data[:,0]), np.min(B_data[:,0]))
    xmax = min(np.max(A_data[:,0]), np.max(B_data[:,0]))
    x_common = np.linspace(xmin, xmax, args.n_points)

    A_rs = resample_xy(A_data, x_common)
    B_rs = resample_xy(B_data, x_common)

    # Métriques
    m = compute_metrics(A_rs[:,1], B_rs[:,1])

    # Erreurs locales
    eps = 1e-12
    err_abs = np.abs(B_rs[:,1] - A_rs[:,1])
    err_pct = err_abs / np.maximum(eps, np.abs(A_rs[:,1])) * 100.0
    comp = np.column_stack([A_rs[:,0], A_rs[:,1], B_rs[:,1], err_abs, err_pct])
    save_csv(os.path.join(args.outdir, "compare_resampled.csv"),
             comp, "x,y_biblio(A),y_modele(B),abs_error,pct_error")
    save_csv(os.path.join(args.outdir, "error_percent.csv"),
             np.column_stack([A_rs[:,0], err_pct]), "x,error_percent")

    # Plots
    unit_x = "x (normalisé)" if limitsA is None or limitsB is None else "x"
    unit_y = "y (normalisé)" if limitsA is None or limitsB is None else "y"
    plot_overlay(os.path.join(args.outdir, "overlay.png"), A_data, B_data, A_rs, B_rs, (unit_x, unit_y))
    plot_errpct(os.path.join(args.outdir, "error_percent.png"), A_rs[:,0], err_pct, unit_x)

    # Console
    print("==> Metrics")
    for k,v in m.items():
        print(f"{k}: {v:.6g}")
    print(f"Résultats sauvegardés dans {args.outdir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-a", required=True, help="Image Biblio")
    ap.add_argument("--img-b", required=True, help="Image Modèle")
    ap.add_argument("--limits-a", nargs=4, type=float, help="Xmin Xmax Ymin Ymax (optionnel)")
    ap.add_argument("--limits-b", nargs=4, type=float, help="Xmin Xmax Ymin Ymax (optionnel)")
    ap.add_argument("--color-a", default="blue")
    ap.add_argument("--color-b", default="blue")
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
