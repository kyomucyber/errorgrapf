#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Affiche les résultats produits par graph.py :
- overlay.png : superposition des courbes
- error_percent.png : erreur relative (%) vs x
- compare_resampled.csv : reconstruit l’erreur % si le PNG manque

Usage :
  python view_results.py --outdir out
  python view_results.py --outdir out --pause 0.5
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def show_image(path: str, title: str):
    img = plt.imread(path)
    plt.figure(title)
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()

def show_error_from_csv(csv_path: str, title: str):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    x = data[:, 0]
    yA = data[:, 1]
    yB = data[:, 2]
    err_pct = np.abs(yB - yA) / np.maximum(1e-12, np.abs(yA)) * 100.0
    plt.figure(title)
    plt.plot(x, err_pct)
    plt.xlabel("x (temps)")
    plt.ylabel("Erreur [%]")
    plt.title("Erreur relative (%) — reconstruite depuis compare_resampled.csv")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    p = argparse.ArgumentParser(description="Affiche overlay + erreur % générés par graph.py")
    p.add_argument("--outdir", default="out", help="Dossier des résultats")
    p.add_argument("--pause", type=float, default=0.0, help="Pause après la 1re fenêtre (s)")
    args = p.parse_args()

    overlay_png = os.path.join(args.outdir, "overlay.png")
    err_png = os.path.join(args.outdir, "error_percent.png")
    comp_csv = os.path.join(args.outdir, "compare_resampled.csv")

    found_any = False

    # Affiche overlay
    if os.path.isfile(overlay_png):
        show_image(overlay_png, "Overlay — Biblio vs Modèle")
        found_any = True
        if args.pause > 0:
            plt.pause(args.pause)
    else:
        print(f"Manquant: {overlay_png}")

    # Affiche erreur %
    if os.path.isfile(err_png):
        show_image(err_png, "Erreur %")
        found_any = True
    elif os.path.isfile(comp_csv):
        show_error_from_csv(comp_csv, "Erreur % (reconstruite)")
        found_any = True
    else:
        print(f"Manquant: {err_png} et {comp_csv}")

    if found_any:
        plt.show()

if __name__ == "__main__":
    main()
