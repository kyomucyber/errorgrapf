#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects

# ---------- utilitaires ----------
def hex_to_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6],16), int(h[2:4],16), int(h[0:2],16))  # B,G,R

def deltaE76_lab(lab_img, lab_ref):
    # lab_img: HxWx3 float32
    d = lab_img - lab_ref.reshape(1,1,3)
    return np.sqrt(np.sum(d*d, axis=2))

def ensure_out(outdir):
    os.makedirs(outdir, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

# ---------- détection ROI ----------
def detect_plot_roi(img_bgr):
    # masque "non-blanc" pour écarter les marges
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 230), (180, 30, 255))
    nonwhite = cv2.bitwise_not(white)
    k = np.ones((7,7), np.uint8)
    nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, k, 2)
    cnts, _ = cv2.findContours(nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w = img_bgr.shape[:2]
        return (0,0,w,h)
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    pad = int(0.02*max(w,h))
    x = max(0, x-pad); y = max(0, y-pad)
    w = min(img_bgr.shape[1]-x, w+2*pad)
    h = min(img_bgr.shape[0]-y, h+2*pad)
    return (x,y,w,h)

# ---------- extraction par couleur ----------
def color_mask(img_bgr, hex_color, dE_thr=22, sat_min=25):
    bgr_ref = np.array(hex_to_bgr(hex_color), dtype=np.uint8).reshape(1,1,3)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(bgr_ref, cv2.COLOR_BGR2LAB).astype(np.float32)[0,0,:]
    dE = deltaE76_lab(lab, lab_ref)
    # évite le texte sombre: impose un minimum de saturation
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]
    mask = (dE <= dE_thr) & (S >= sat_min)
    return mask.astype(np.uint8)*255

def pick_longest_component(bin_mask, min_span_ratio=0.5):
    # supprime les petites composantes puis garde celle qui couvre le plus en X
    lbl = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    n, stats = lbl[0], lbl[2]
    if n<=1:
        return bin_mask
    h, w = bin_mask.shape
    best = None
    best_span = -1
    out = np.zeros_like(bin_mask)
    for i in range(1, n):
        x,y,ww,hh,area = stats[i]
        if area < 100:  # petit bruit
            continue
        span = ww / max(1,w)
        if span > best_span:
            best_span = span
            best = i
    if best is None or best_span < min_span_ratio:
        # fallback: garder tout, on comblera ensuite
        return bin_mask
    out[lbl[1]==best] = 255
    return out

def horizontal_bridge(mask, it=2, width=9):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    return cv2.dilate(mask, k, iterations=it)

def skeleton_from_mask(mask):
    sk = skeletonize((mask>0).astype(bool))
    # nettoyage des brindilles
    sk = remove_small_objects(sk, 30, connectivity=2)
    return sk.astype(np.uint8)*255

def trace_centerline(sk):
    # pour chaque colonne, prend la médiane des pixels actifs; comble par interpolation
    h, w = sk.shape
    xs, ys = [], []
    for u in range(w):
        col = np.where(sk[:,u]>0)[0]
        if col.size>0:
            xs.append(u); ys.append(int(np.median(col)))
    if len(xs) < 5:
        return None
    xs = np.array(xs); ys = np.array(ys)
    # rééchantillonnage régulier en X
    x_full = np.arange(0, w)
    y_full = np.interp(x_full, xs, ys)
    return np.column_stack([x_full, y_full])

def map_to_data(points_xy, roi, limits=None):
    x0,y0,w,h = roi
    u = points_xy[:,0].astype(np.float64)
    v = points_xy[:,1].astype(np.float64)
    xn = (u - 0) / max(1.0, w-1)
    yn = 1.0 - (v - 0) / max(1.0, h-1)
    if limits is None:
        return np.column_stack([xn, yn])
    xmin,xmax,ymin,ymax = limits
    xd = xmin + xn*(xmax-xmin)
    yd = ymin + yn*(ymax-ymin)
    return np.column_stack([xd, yd])

# ---------- pipeline ----------
def extract_curve(img_path, hex_color, limits=None, outdir="out", debug=False):
    ensure_out(outdir)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    x,y,w,h = detect_plot_roi(img)
    roi_img = img[y:y+h, x:x+w].copy()

    if debug:
        dbg = img.copy()
        cv2.rectangle(dbg, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.imwrite(os.path.join(outdir,"debug_roi.png"), dbg)

    m0 = color_mask(roi_img, hex_color)
    m1 = pick_longest_component(m0, min_span_ratio=0.35)
    m2 = horizontal_bridge(m1, it=2, width=max(7, w//120))
    sk = skeleton_from_mask(m2)
    path = trace_centerline(sk)
    if path is None:
        raise RuntimeError("Courbe non trouvée après squelettisation.")

    # exports debug
    if debug:
        cv2.imwrite(os.path.join(outdir,"debug_mask_raw.png"), m0)
        cv2.imwrite(os.path.join(outdir,"debug_mask_kept.png"), m1)
        cv2.imwrite(os.path.join(outdir,"debug_mask_bridged.png"), m2)
        cv2.imwrite(os.path.join(outdir,"debug_skeleton.png"), sk)

    # mapping vers données
    data = map_to_data(path, (0,0,w,h), limits)

    # overlay
    overlay = roi_img.copy()
    for i in range(path.shape[0]-1):
        p1 = (int(path[i,0]),   int(path[i,1]))
        p2 = (int(path[i+1,0]), int(path[i+1,1]))
        cv2.line(overlay, p1, p2, (0,255,0), 1)
    cv2.imwrite(os.path.join(outdir,"overlay_path_on_roi.png"), overlay)

    # plot data
    plt.figure(figsize=(6,4))
    plt.plot(data[:,0], data[:,1])
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Courbe extraite")
    savefig(os.path.join(outdir,"curve_extracted.png"))

    np.savetxt(os.path.join(outdir,"series_extracted.csv"), data, delimiter=",", header="x,y", comments="")

    return data

# ---------- comparaison simple (optionnel) ----------
def resample(xy, x_new):
    x=xy[:,0]; y=xy[:,1]; idx=np.argsort(x)
    return np.column_stack([x_new, np.interp(x_new, x[idx], y[idx])])

def compare_two(series_a, series_b, outdir="out"):
    xmin = max(series_a[:,0].min(), series_b[:,0].min())
    xmax = min(series_a[:,0].max(), series_b[:,0].max())
    xs = np.linspace(xmin, xmax, 400)
    A = resample(series_a, xs); B = resample(series_b, xs)
    err = np.abs(B[:,1]-A[:,1])
    pct = err/np.maximum(1e-12, np.abs(A[:,1]))*100.0
    np.savetxt(os.path.join(outdir,"compare_resampled.csv"),
               np.column_stack([xs, A[:,1], B[:,1], err, pct]),
               delimiter=",", header="x,y_a,y_b,abs_error,pct_error", comments="")
    plt.figure(figsize=(6,4))
    plt.plot(xs, pct)
    plt.xlabel("x"); plt.ylabel("Erreur [%]")
    plt.title("Erreur relative (%)")
    savefig(os.path.join(outdir,"error_percent.png"))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="chemin image")
    ap.add_argument("--hex", default="#1F497D", help="couleur cible en hex (ex: #1F497D)")
    ap.add_argument("--limits", nargs=4, type=float, help="xmin xmax ymin ymax (optionnel)")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--debug", action="store_true")
    # option comparaison 2e image
    ap.add_argument("--img2", help="deuxième image (optionnel)")
    ap.add_argument("--hex2", help="couleur cible image 2")
    ap.add_argument("--limits2", nargs=4, type=float)
    args = ap.parse_args()

    ensure_out(args.outdir)
    A = extract_curve(args.img, args.hex, args.limits, args.outdir, args.debug)

    if args.img2:
        hex2 = args.hex2 if args.hex2 else args.hex
        B = extract_curve(args.img2, hex2, args.limits2, args.outdir, args.debug)
        compare_two(A, B, args.outdir)

if __name__ == "__main__":
    main()
