<img width="460" height="94" alt="image" src="https://github.com/user-attachments/assets/c9655aa7-cef6-4576-a4b9-887fa0d066cd" /><img width="622" height="94" alt="image" src="https://github.com/user-attachments/assets/e0bac7b5-8343-4a74-818a-c79f75ba1098" /># Interactive Spatial Models in Ecology and Evolutionary Biology

This collection of interactive teaching demos illustrates classic models in
spatial ecology, evolutionary game theory, and movement ecology. Each script
is a standalone, self-contained simulation with a GUI for exploring how
model parameters shape emergent spatial patterns.

They support theoretical lectures delivered as part of the course 'Spatial mathematical modeling'
(SoSe2026; 5.15.7541) at the Master's in Environmental Modeling at the University of Oldenburg. 


## Contents

| Script | Model | Reference |
|---|---|---|
| `python/spatial-logistic.py` | Off-lattice logistic model, Gaussian competition kernel | Surendran, Pinto-Ramos, Menezes & Martinez-Garcia, *Physica D* 477 (2025) 134703 |
| `python/spatial-logistic-tophat.py` | Off-lattice logistic model, top-hat competition kernel | SSurendran, Pinto-Ramos, Menezes & Martinez-Garcia, *Physica D* 477 (2025) 134703 |
| `python/cp1d.py` | On-lattice logistic model (1D contact process) | Classic birth-death process on a ring |
| `python/colicin.py` | Allelopathy (colicin producer/sensitive) lattice model | Durrett & Levin, *J. Theor. Biol.* 185 (1997) 165-171 |
| `python/rps-reichenbach.py` | Spatial rock-paper-scissors with mobility | Reichenbach, Mobilia & Frey, *Nature* 448 (2007) 1046-1049 |
| `python/kerr-rps.py` | C-S-R (colicinogenic/sensitive/resistant) rock-paper-scissors | Kerr, Riley, Feldman & Bohannan, *Nature* 418 (2002) 171-174 |
| `html/ssf.html` | Step-selection function (movement ecology) simulator | Browser-based, no dependencies |

## Requirements

The Python scripts need:

```
numpy
matplotlib
```

Tkinter is used for the GUI and ships with most Python installations (on
Linux you may need `sudo apt install python3-tk`).

Install with:

```bash
pip install -r requirements.txt
```

## Running

Each Python script is standalone:

```bash
python python/spatial-logistic.py
```

The HTML file needs no installation — just open `html/ssf.html` in a browser.

## License

Add a license (e.g. MIT) if you want others to reuse these freely — see
[choosealicense.com](https://choosealicense.com).
