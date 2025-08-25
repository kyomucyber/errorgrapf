#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
install_requirements.py
Installe toutes les dépendances avec des versions compatibles (Python 3.12+ conseillé).
"""

import subprocess, sys

# --- Versions stables et compatibles ---
REQUIRED_PACKAGES = {
    "numpy": ">=2.1,<3",
    "pandas": ">=2.2,<3",
    "scipy": ">=1.12,<2",
    "scikit-image": ">=0.23,<1",
    "opencv-python": ">=4.9,<5",
    "matplotlib": ">=3.8,<4",
    "Pillow": ">=10,<11",
    "pytesseract": ">=0.3.10,<0.4",
}

def run(cmd):
    print(f"➡️  {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"❌ Erreur lors de la commande: {cmd}")
        sys.exit(1)

def main():
    print("=== Installation des dépendances ===")
    print(f"Version Python détectée: {sys.version.split()[0]}")
    
    # Upgrade pip
    run(f"{sys.executable} -m pip install --upgrade pip")

    # Installer chaque package avec la bonne version
    for pkg, ver in REQUIRED_PACKAGES.items():
        run(f"{sys.executable} -m pip install \"{pkg}{ver}\"")

    print("\n✅ Installation terminée avec succès !")

if __name__ == "__main__":
    main()
