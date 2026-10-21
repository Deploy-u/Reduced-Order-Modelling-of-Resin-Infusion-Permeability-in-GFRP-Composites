import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

# Set publication-quality styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300
})

# ==============================================================================
# 1. PHYSICAL MODELS (DARCY FLOW & GEBART MICROMECHANICAL PERMEABILITY)
# ==============================================================================
def gebart_permeability_axial(Vf, Rf=8.5e-6, c=53.0):
    """Gebart (1992) axial permeability for hexagonal fibre packing (m^2)."""
    return (8.0 * Rf**2 / c) * ((1.0 - Vf)**3 / (Vf**2))

def gebart_permeability_transverse(Vf, Rf=8.5e-6):
    """Gebart (1992) transverse permeability for hexagonal fibre packing (m^2)."""
    C1 = 16.0 / (9.0 * np.pi * np.sqrt(6.0))
    Vf_max = np.pi / (2.0 * np.sqrt(3.0))
    return C1 * (Rf**2) * ((np.sqrt(Vf_max / Vf) - 1.0)**2.5)

def analytical_flow_front(t, K, dP, phi, mu):
    """Analytical 1D Darcy flow front position x_f(t) in meters."""
    return np.sqrt((2.0 * K * dP * t) / (phi * mu))

def analytical_fill_time(L, K, dP, phi, mu):
    """Analytical mould filling time t_fill for length L in seconds."""
    return (phi * mu * L**2) / (2.0 * K * dP)

def numerical_rk4_flow_front(L, K, dP, phi, mu, n_steps=500):
    """Numerical integration of dx_f/dt = (K*dP)/(phi*mu*x_f) using RK4."""
    t_max = analytical_fill_time(L, K, dP, phi, mu)
    t = np.linspace(1e-3, t_max, n_steps)
    dt = t[1] - t[0]
    x_num = np.zeros_like(t)
    x_num[0] = analytical_flow_front(t[0], K, dP, phi, mu) # Avoid x=0 singularity
    
    coeff = (K * dP) / (phi * mu)
    for i in range(len(t) - 1):
        k1 = coeff / x_num[i]
        k2 = coeff / (x_num[i] + 0.5 * dt * k1)
        k3 = coeff / (x_num[i] + 0.5 * dt * k2)
        k4 = coeff / (x_num[i] + dt * k3)
        x_num[i+1] = x_num[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return t, x_num

# ==============================================================================
# 2. BASELINE PARAMETERS
# ==============================================================================
L_base = 0.50          # Mould length (m)
Vf_base = 0.50         # Fibre volume fraction (-)
phi_base = 1.0 - Vf_base
Rf_base = 8.5e-6       # E-glass fibre radius (m)
K_base = gebart_permeability_axial(Vf_base, Rf_base) # ~1.36e-10 m^2
mu_base = 0.25         # Epoxy viscosity (Pa.s)
dP_base = 1.0e5        # Vacuum infusion pressure 1 bar (Pa)

# ==============================================================================
# 3. FIGURE 1: FLOW-FRONT PROGRESSION & NUMERICAL VALIDATION
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4.8))
pressures_bar = [0.5, 1.0, 2.0, 3.0]
colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']

for P_bar, col in zip(pressures_bar, colors):
    dP = P_bar * 1e5
    t_fill = analytical_fill_time(L_base, K_base, dP, phi_base, mu_base)
    t_arr = np.linspace(0, t_fill, 300)
    x_arr = analytical_flow_front(t_arr, K_base, dP, phi_base, mu_base)
    ax.plot(t_arr / 60.0, x_arr * 100.0, color=col, lw=2.2,
            label=f'$\\Delta P = {P_bar:.1f}$ bar ($t_{{fill}} = {t_fill/60:.1f}$ min)')

# Overlay RK4 numerical verification on baseline (1.0 bar)
t_rk4, x_rk4 = numerical_rk4_flow_front(L_base, K_base, dP_base, phi_base, mu_base, n_steps=25)
ax.scatter(t_rk4 / 60.0, x_rk4 * 100.0, color='black', zorder=5, s=22,
           label='RK4 Numerical (1.0 bar)')

ax.axhline(L_base * 100.0, color='gray', linestyle='--', lw=1.2, label='Mould Exit ($50$ cm)')
ax.set_xlim(left=0)
ax.set_ylim(0, 55)
ax.set_xlabel('Infusion Time $t$ (minutes)')
ax.set_ylabel('Flow-Front Position $x_f(t)$ (cm)')
ax.set_title('1D Darcy Flow-Front Progression & Numerical Verification')
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('fig1_flow_front.png')
plt.show()

# ==============================================================================
# 4. FIGURE 2: MICROMECHANICAL-PROCESSING TRADE-OFF (Vf vs K and t_fill)
# ==============================================================================
Vf_range = np.linspace(0.35, 0.68, 200)
phi_range = 1.0 - Vf_range
K_ax = gebart_permeability_axial(Vf_range, Rf_base)
K_tr = gebart_permeability_transverse(Vf_range, Rf_base)
t_fill_ax = analytical_fill_time(L_base, K_ax, dP_base, phi_range, mu_base) / 60.0 # minutes

fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
ax2 = ax1.twinx()

l1 = ax1.semilogy(Vf_range * 100, K_ax, 'b-', lw=2.2, label=r'Axial Permeability $K_{\parallel}$ (Gebart)')
l2 = ax1.semilogy(Vf_range * 100, K_tr, 'b--', lw=2.0, label=r'Transverse Permeability $K_{\perp}$ (Gebart)')
l3 = ax2.plot(Vf_range * 100, t_fill_ax, 'r-', lw=2.5, label=r'Axial Filling Time $t_{fill}$ ($\Delta P = 1$ bar)')

ax1.set_xlabel('Fibre Volume Fraction $V_f$ (%)')
ax1.set_ylabel(r'Preform Permeability $K$ ($\mathrm{m^2}$)', color='b')
ax2.set_ylabel('Mould Filling Time $t_{fill}$ (minutes)', color='r')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='r')
ax1.grid(True, which='both', alpha=0.25)

lines = l1 + l2 + l3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')
plt.title('Effect of Fibre Volume Fraction on Permeability & Infusion Time')
plt.tight_layout()
plt.savefig('fig2_vf_permeability_tradeoff.png')
plt.show()

# ==============================================================================
# 5. FIGURE 3: 2D PARAMETRIC SENSITIVITY CONTOUR (VISCOSITY vs PERMEABILITY)
# ==============================================================================
mu_grid = np.linspace(0.10, 0.80, 150)
K_grid = np.logspace(-11, -9.5, 150)
MU, KK = np.meshgrid(mu_grid, K_grid)
T_FILL_MIN = analytical_fill_time(L_base, KK, dP_base, phi_base, MU) / 60.0

fig, ax = plt.subplots(figsize=(7, 5))
cp = ax.contourf(MU, KK, np.log10(T_FILL_MIN), levels=20, cmap='viridis')
cbar = fig.colorbar(cp, ax=ax)
cbar.set_label(r'$\log_{10}(t_{fill} \text{ [minutes]})$')

# Iso-time contour for a typical 30-minute epoxy gelation window
cs = ax.contour(MU, KK, T_FILL_MIN, levels=[10, 30, 60, 120], colors='white', linewidths=1.5)
ax.clabel(cs, fmt='%d min', fontsize=9)

ax.set_yscale('log')
ax.set_xlabel(r'Resin Viscosity $\mu$ ($\mathrm{Pa\cdot s}$)')
ax.set_ylabel(r'Preform Permeability $K$ ($\mathrm{m^2}$)')
ax.set_title('Process Window Map: Filling Time vs. Viscosity & Permeability')
plt.tight_layout()
plt.savefig('fig3_parametric_contour.png')
plt.show()

print("All 3 figures successfully generated and displayed.")