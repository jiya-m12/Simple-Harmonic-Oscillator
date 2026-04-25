# Quantum Algorithm for the Simple Harmonic Oscillator

**Team Entangled Ancillas** — Q-volution Hackathon 2026 (Girls in Quantum) · Track C: Harmonic Oscillator

---

## Overview

This project implements a quantum algorithm to simulate the **Simple Harmonic Oscillator (SHO)** using the **Linear Combination of Unitaries (LCU)** framework, built on [Classiq](https://www.classiq.io/).

Inspired by [Tao Xin et al. (2020)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032307), we constructed quantum circuits to compute **kinetic and potential energy over time** and verified that **total energy is conserved** throughout the simulation.

We also traced the same mathematical structure to broader real-world applications, including **protein folding dynamics** and **heat conduction**.

---

## Repository Structure

```
├── notebook/
│   ├── quantum_algorithm_for_sho.ipynb   # Main notebook: problem statement, implementation & analysis
│   ├── quantum.py                         # Core algorithm (4 analysis methods)
│   ├── requirements.txt                   # Python dependencies
│   ├── bound_analysis.png
│   ├── energy-analysis.png
│   ├── k_accuracy.png
│   ├── k_resources.png
│   ├── postselection_rate.png
│   └── shots_trajectories.png
├── entangled_ancillas_hackathon_summary.pdf   # 1-page project summary
├── quantum_algorithm_for_sho.pdf             # Full notebook as PDF
└── README.md
```

---

## The Algorithm

`quantum.py` contains four analysis methods:

| Method | What it does |
|---|---|
| `energy_analysis` | Computes kinetic and potential energy over time and verifies conservation |
| `bound_analysis` | Analyzes how boundary conditions affect the simulation |
| `k_analysis` | Examines accuracy as a function of the LCU parameter *k* |
| `n_shots_analysis` | Studies how the number of measurement shots affects results |

> ⚠️ Each method can take a few minutes to run. They are **commented out by default** — uncomment and run one at a time.

---

## How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/jiya-m12/Simple-Harmonic-Oscillator.git
cd Simple-Harmonic-Oscillator/notebook
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Authenticate with Classiq** *(first time only)*
```python
# Uncomment the authentication line at the top of quantum.py and run it once
```

**5. Open the notebook**
```bash
jupyter notebook quantum_algorithm_for_sho.ipynb
```

**6. Run the algorithm directly** *(optional)*
```bash
python3 quantum.py
```

---

## References

- Tao Xin et al. (2020). *Quantum algorithm for solving linear differential equations: Theory and experiment*(https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032307)
- [Classiq Platform](https://www.classiq.io/)
- [Q-volution Hackathon 2026 — Girls in Quantum](https://girlsinquantum.com/)
