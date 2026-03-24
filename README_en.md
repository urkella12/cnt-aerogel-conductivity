# CNT-Aerogel Composite Electrical Conductivity Model

🔬 **Computational model for electrical conductivity in carbon nanotube nanocomposites**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

[🇷🇺 Русская версия](README.md)

---

## 📋 Description

Computational model for investigating electrical conductivity of CNT composites.

**Methods:**
- 3D geometric modeling
- Percolation theory (graph theory)
- Kirchhoff's method for conductivity
- Tunneling between CNTs
- Monte Carlo statistical analysis

**Key Results:**
- Percolation threshold: **φ_c = 0.27 ± 0.07%**
- Power law: **σ ∝ (φ - φ_c)^1.86** (R² = 0.9989)
- Calibrated on 250+ simulations

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/cnt-aerogel-conductivity.git
cd cnt-aerogel-conductivity

# Install dependencies
pip install numpy pandas networkx scipy openpyxl

# Run simulation
cd current
python ФИНАЛ.py
```

---

## 💻 Usage

### Single Simulation
```bash
cd current
python ФИНАЛ.py
```

### Batch Mode
```bash
cd current
python ФИНАЛЬНЫЙ_БАЧ_.py

# Choose scenario:
# 1 - CNT concentration [8 points, ~15 min]
# 2 - Geometry [11 points, ~25 min]
# 4 - Full calibration [22 points, ~45 min]
# 5 - Detailed analysis [250 sims, ~4-5 hours] ⭐
```

### Code Example
```python
import sys
sys.path.append('./current')
from ФИНАЛ import EnhancedNanotubeSimulator

sim = EnhancedNanotubeSimulator()
sim.num_tubes = 1000
sim.tube_length = 250.0  # nm
sim.generate_all_objects()
sim.calculate_conductivity()

print(f"σ = {sim.conductivity_results['sigma_effective']:.2e} S/m")
```

---

## 📊 Main Results

| Parameter | Value | Description |
|-----------|-------|-------------|
| φ_c | 0.27 ± 0.07% | Percolation threshold |
| t | 1.86 ± 0.05 | Critical exponent |
| R² | 0.9989 | Power law fit quality |
| α | 0.011 nm⁻¹ | Length dependence |

---

## 🎓 Applicability

### ✅ Valid for:
- Dense matrix composites (ρ > 0.2 g/cm³)
- Volume fractions φ = 0.1-5%
- Tube lengths L = 100-500 nm

### ❌ Not valid for:
- Aerogels (porosity >90%)
- Macroscopic samples (>10 μm)
- Very long CNTs (L >> 1 μm)

---

## 📄 License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**

- ✅ Free to use (non-commercial)
- ✅ Can modify
- ⚠️ **MUST** credit author
- ⚠️ **NO** commercial use
- ⚠️ Derivatives under same license

See [LICENSE](LICENSE) for details.

### Citation
```
[Your Name]. (2025). CNT-Aerogel Composite Electrical Conductivity Model.
GitHub: https://github.com/[username]/cnt-aerogel-conductivity
```

---

## 👨‍💻 Author

**Bachelor's Thesis**  
Grichenuk U.U.  
MUCTR, 2025

📧 urkella1@gmail.com
🔗 [@username](https://github.com/urkella12)

---

**⭐ Star this project if you find it useful!**
