import numpy as np

import numpy as np
import matplotlib.pyplot as plt

def f_config3_modified(x, kt, D1, D2, k, v, x_l, blend=0.0):
    """
    Force-déplacement pour le modèle équivalent:
      - Zone 1 (x <= x_l): f = k_t * x
      - Zone 2 (x >  x_l): f = (D1*D2 / (D2*exp(-(k/D1)*(x/v)) + D1)) * x

    Paramètres
    ----------
    x : array_like
        Déplacement [mm] (peut être scalaire ou vecteur).
    kt : float
        Pente linéaire initiale (N/mm).
    D1 : float
        Coefficient d’amortissement du branchement parallèle (N·s/mm).
    D2 : float
        Coefficient d’amortissement en série (N·s/mm).
    k : float
        Raideur du ressort dans le branchement parallèle (N/mm).
    v : float
        Vitesse de compression utilisée pour l’identification/simu (mm/s).
    x_l : float
        Déplacement de transition entre zone linéaire et zone non-linéaire (mm).
    blend : float, optional
        Demi-largeur d’une zone lissée autour de x_l pour un raccord C^0/C^1 (mm).
        0.0 désactive le lissage.

    Retour
    ------
    f : ndarray
        Force correspondante [N] pour chaque x.
    """
    x = np.asarray(x, dtype=float)

    # non-linéaire
    expo  = np.exp(-(k / D1) * (x / v))
    denom = D2 * expo + D1
    f_nl  = (D1 * D2 / denom) * x

    # linéaire
    f_lin = kt * x

    if blend <= 0.0:
        f = np.where(x <= x_l, f_lin, f_nl)
    else:
      
        x1, x2 = x_l - blend, x_l + blend
        w = np.zeros_like(x)
        in_blend = (x > x1) & (x < x2)
        
        w[in_blend] = 0.5 * (1.0 - np.cos(np.pi * (x[in_blend] - x1) / (2.0 * blend)))
        w[x >= x2] = 1.0
        f = (1.0 - w) * f_lin + w * f_nl

    f = np.where(f < 1e-12, 0.0, f)
    # 
    if x[0] != 0.0:
        x = np.insert(x, 0, 0.0)
        f = np.insert(f, 0, 0.0)
    else:
        f[0] = 0.0

    return f


if __name__ == "__main__":
    # Paramètres  (SOC=0, v=100 mm/s)
    kt = 1345.7
    D1 = 108.2
    D2 = 1.29e4
    k  = 6772.0
    v  = 100.0
    x_l = 4.0
    blend = 0.2

    x = np.linspace(0.0, 7.0, 400)  # mm
    f = f_config3_modified(x, kt, D1, D2, k, v, x_l, blend=blend)

    #  σ-ε 
    R = 8.8   # mm
    L = 58.0  # mm
    eps_av = x / (2.0 * R)
    eps_safe = np.where(eps_av == 0, 1e-12, eps_av)
    sigma_av = f / (np.pi * R * L * eps_safe)

    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(x, f, linewidth=2)
    ax.axvline(x_l, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title("Configuration 3 — Modèle ressort–amortisseur")
    ax.set_xlabel("Déplacement x (mm)")
    ax.set_ylabel("Force f(x) (N)")
    ax.grid(True, alpha=0.3)


    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0, y=0) 

    plt.tight_layout()
    plt.show()

   
   
