#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from skimage.morphology import skeletonize, remove_small_objects
import pytesseract

# ========== utils ==========
def _safe_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def _largest_lines_hv(edges, min_len_ratio=0.4):
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=int(min(h,w)*min_len_ratio),
                            maxLineGap=15)
    if lines is None: return [], []
    horiz, vert = [], []
    for x1,y1,x2,y2 in lines[:,0,:]:
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dy <= 2 and dx >= dy:   # horizontal
            horiz.append((x1,y1,x2,y2))
        elif dx <= 2 and dy > dx:  # vertical
            vert.append((x1,y1,x2,y2))
    # garder la plus longue pour chaque orientation
    if horiz:
        horiz.sort(key=lambda L: abs(L[2]-L[0]), reverse=True)
        horiz = [horiz[0]]
    if vert:
        vert.sort(key=lambda L: abs(L[3]-L[1]), reverse=True)
        vert = [vert[0]]
    return horiz, vert

def detect_axes(img_bgr) -> Dict:
    """Détecte l’axe X (horizontal) et l’axe Y (vertical). Renvoie leurs lignes et un masque ROI plot."""
    gray = _safe_gray(img_bgr)
    # netteté sur les traits
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blur, 60, 160)
    hlines, vlines = _largest_lines_hv(edges)
    H, W = gray.shape
    xaxis = hlines[0] if hlines else (0, H-1, W-1, H-1)
    yaxis = vlines[0] if vlines else (0, 0, 0, H-1)

    # approx ROI: à droite de yaxis et au-dessus de xaxis
    x0 = min(yaxis[0], yaxis[2]); x0 = max(0, min(x0+4, W-1))
    y0 = min(xaxis[1], xaxis[3]); y0 = max(0, min(y0-4, H-1))
    roi = (x0, 0, W-x0, y0) if y0>0 else (x0, 0, W-x0, H)  # fallback
    return {"xaxis": xaxis, "yaxis": yaxis, "roi": roi}

# ========== OCR ticks ==========
def _ocr_text(patch_bgr) -> str:
    # binarisation douce
    gray = _safe_gray(patch_bgr)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # config OCR seulement chiffres + signes
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-+eE"
    txt = pytesseract.image_to_string(th, config=cfg)
    return txt.strip()

def read_ticks(img_bgr, axes: Dict, search_pad=36) -> Dict:
    """Lit quelques graduations sur X et Y avec OCR pour déduire l’échelle."""
    H, W = img_bgr.shape[:2]
    x1,y1,x2,y2 = axes["xaxis"]
    xv1,yv1,xv2,yv2 = axes["yaxis"]

    # bandes de recherche pour labels
    # X labels: sous l’axe X
    y_top = min(y1,y2)
    y_a = min(H-1, y_top + 6)
    y_b = min(H-1, y_a + search_pad)
    band_x = img_bgr[y_a:y_b, 0:W]

    # Y labels: à gauche de l’axe Y
    x_left = min(xv1,xv2)
    x_a = max(0, x_left - search_pad)
    x_b = max(1, x_left - 2)
    band_y = img_bgr[0:H, x_a:x_b]

    # segmenter composantes et OCR quelques patches centrés sur composantes larges
    def _find_labels(band, axis='x'):
        gray = _safe_gray(band)
        inv = cv2.bitwise_not(gray)
        th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, 1)
        cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand=[]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < 60: continue
            if h<8 or w<6: continue
            cand.append((x,y,w,h))
        # prendre jusqu’à 6 labels triés
        cand.sort(key=lambda b: b[0] if axis=='x' else b[1])
        cand = cand[:6]
        out=[]
        for (x,y,w,h) in cand:
            pad=3
            xa=max(0,x-pad); ya=max(0,y-pad)
            xb=min(band.shape[1], x+w+pad); yb=min(band.shape[0], y+h+pad)
            patch = band[ya:yb, xa:xb]
            txt = _ocr_text(patch)
            if txt:
                # coordonnées du centre en image globale
                if axis=='x':
                    cx = xa + (xb-xa)//2
                    cy = y_a + ya + (yb-ya)//2
                else:
                    cx = x_a + xa + (xb-xa)//2
                    cy = ya + (yb-ya)//2
                out.append((txt, cx, cy))
        return out

    labels_x = _find_labels(band_x, 'x')
    labels_y = _find_labels(band_y, 'y')

    return {"labels_x": labels_x, "labels_y": labels_y}

def _parse_numeric(s: str) -> Optional[float]:
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except:
        # formes type 1e3 ou 1E+03 déjà gérées, on tente un nettoyage
        import re
        m = re.findall(r"[-+]?[\d]*\.?[\d]+(?:[eE][-+]?\d+)?", s)
        if not m: return None
        try: return float(m[0])
        except: return None

def build_scale_from_ticks(axes: Dict, ticks: Dict, roi: Tuple[int,int,int,int]) -> Optional[Tuple[Tuple[float,float],Tuple[float,float]]]:
    """Construit l’échelle (xmin,xmax),(ymin,ymax) depuis quelques labels OCR."""
    x0, y0, w, h = roi
    # X: on collecte (pixel_x → valeur)
    xs = []
    for txt, cx, cy in ticks.get("labels_x", []):
        val = _parse_numeric(txt)
        if val is None: continue
        # position x normalisée [0..1] dans ROI
        u = (cx - x0) / max(1.0, w-1)
        xs.append((u, val))
    ys = []
    for txt, cx, cy in ticks.get("labels_y", []):
        val = _parse_numeric(txt)
        if val is None: continue
        v = (cy - (y0)) / max(1.0, h-1)  # v croît vers le bas
        yn = 1.0 - v                     # normaliser en sens mathématique
        ys.append((yn, val))
    if len(xs) >= 2 and len(ys) >= 2:
        # régression linéaire simple val = a*u + b
        def fit_ab(pairs):
            U = np.array([p[0] for p in pairs], dtype=float)
            V = np.array([p[1] for p in pairs], dtype=float)
            A = np.vstack([U, np.ones_like(U)]).T
            a,b = np.linalg.lstsq(A, V, rcond=None)[0]
            return a,b
        ax, bx = fit_ab(xs)  # x_val = ax*u + bx
        ay, by = fit_ab(ys)  # y_val = ay*yn + by
        # bornes ROI: u=0→xmin, u=1→xmax ; yn=0→ymin, yn=1→ymax
        xmin = ax*0.0 + bx; xmax = ax*1.0 + bx
        ymin = ay*0.0 + by; ymax = ay*1.0 + by
        return (xmin, xmax), (ymin, ymax)
    return None

# ========== segmentation courbes ==========
def segment_curves_kmeans(roi_bgr, k=3):
    """K-means couleur pour séparer fond/axes/1-2 courbes. Retourne labels et centres BGR."""
    Z = roi_bgr.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(roi_bgr.shape[:2])
    centers = centers.astype(np.uint8)  # BGR
    return labels, centers

def pick_curve_label(labels, centers):
    """Heuristique: rejette le centre le plus clair (fond), et renvoie le plus saturé restant."""
    hsv = cv2.cvtColor(centers.reshape(1,-1,3), cv2.COLOR_BGR2HSV)[0]
    # rejeter fond très clair
    vals = hsv[:,2]
    bg_idx = int(np.argmax(vals))
    cand = [i for i in range(len(centers)) if i!=bg_idx]
    if not cand: cand=[bg_idx]
    sats = [(i, hsv[i,1]) for i in cand]
    sats.sort(key=lambda t: t[1], reverse=True)
    return sats[0][0]

def mask_from_label(labels, label_id):
    m = (labels==label_id).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return m

def skeleton_path(mask):
    sk = skeletonize(mask>0)
    sk = remove_small_objects(sk, 40, connectivity=2)
    sk8 = (sk.astype(np.uint8))*255
    # centerline par colonne
    h,w = sk8.shape
    xs, ys = [], []
    for u in range(w):
        col = np.where(sk8[:,u]>0)[0]
        if col.size>0:
            xs.append(u); ys.append(int(np.median(col)))
    if len(xs)<5: return None
    xs = np.array(xs); ys=np.array(ys)
    x_full = np.arange(0,w)
    y_full = np.interp(x_full, xs, ys)
    return np.column_stack([x_full, y_full])

# ========== pipeline principal ==========
def extract_plot(img_bgr) -> Dict:
    axes = detect_axes(img_bgr)
    x0,y0,w,h = axes["roi"]
    roi = img_bgr[y0:y0+h, x0:x0+w].copy()

    # multi-courbes par K-means
    labels, centers = segment_curves_kmeans(roi, k=3)
    label_curve = pick_curve_label(labels, centers)
    mask = mask_from_label(labels, label_curve)

    # OCR ticks pour l’échelle
    ticks = read_ticks(img_bgr, axes)
    scales = build_scale_from_ticks(axes, ticks, (x0,y0,w,h))  # ((xmin,xmax),(ymin,ymax)) ou None

    # squelette → chemin
    path = skeleton_path(mask)
    if path is None:
        raise RuntimeError("Courbe non trouvée")

    # mapping pixel→data
    u = path[:,0] / max(1.0, w-1)
    v = path[:,1] / max(1.0, h-1)
    yn = 1.0 - v
    if scales is None:
        xy = np.column_stack([u, yn])  # normalisés si OCR insuffisant
    else:
        (xmin,xmax),(ymin,ymax)=scales
        xval = xmin + u*(xmax-xmin)
        yval = ymin + yn*(ymax-ymin)
        xy = np.column_stack([xval, yval])

    return {
        "axes": axes,
        "ticks": ticks,
        "scales": scales,   # None si OCR insuffisant
        "mask": mask,
        "path_px": path,    # pixels ROI
        "series": xy,       # normalisé ou physique
        "roi": (x0,y0,w,h)
    }
