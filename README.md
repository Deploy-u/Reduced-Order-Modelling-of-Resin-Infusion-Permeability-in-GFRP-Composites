# Reduced-Order Modelling of Resin Infusion in GFRP Composites

Python simulation of 1D resin flow in glass-fibre composites. Uses Darcy's law and Gebart micromechanics to link fibre packing to permeability and filling time. Includes numerical solvers, process window mapping, and Capillary Number analysis to predict manufacturing void defects.

## 📌 Project Overview
This repository contains a computational study on the liquid composite moulding (LCM) process. It models the infiltration of liquid thermoset resin into a dry porous E-glass fibre preform. By coupling macroscopic fluid dynamics with microstructural parameters, the simulation explores the fundamental trade-off between composite structural stiffness (driven by high fibre volume fraction) and processability (driven by permeability and filling time).

## 🚀 Features
- **1D Darcy Flow Simulation:** Calculates flow-front progression $x_f(t)$ and mould-filling time.
- **Gebart Micromechanics:** Couples fibre volume fraction ($V_f$) to axial and transverse preform permeability ($K$).
- **Numerical Validation:** Validates analytical closed-form solutions against a custom 4th-order Runge-Kutta (RK4) ODE solver.
- **Process Window Mapping:** Generates 2D parametric contour maps over viscosity and permeability spaces to identify optimal processing windows prior to resin gelation.
- **Defect Prediction (Void Analysis):** Calculates the Modified Capillary Number ($Ca^*$) to predict spatial risks of intra-tow micro-voids and inter-tow macro-voids.

## 🧮 Governing Equations

**1. 1D Darcy Flow Front Progression:**
$$x_f(t) = \sqrt{\frac{2 K \Delta P \, t}{\phi \mu}}$$

**2. Gebart Axial Permeability (Hexagonal Packing):**
$$K_{\parallel} = \frac{8 R_f^2}{53} \frac{(1 - V_f)^3}{V_f^2}$$

**3. Modified Capillary Number ($Ca^*$):**
$$Ca^* = \frac{\mu \, v_f}{\gamma \cos\theta}$$

## 🛠️ Installation & Usage
### Prerequisites
- Python 3.8+
- NumPy
- Matplotlib

### Running the Simulation
Clone the repository and run the main simulation scripts:
```bash
git clone [https://github.com/yourusername/GFRP-Resin-Infusion-Model.git](https://github.com/yourusername/GFRP-Resin-Infusion-Model.git)
cd GFRP-Resin-Infusion-Model
python lcm_darcy_simulation.py
