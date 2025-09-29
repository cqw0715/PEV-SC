# PEV-SC
PEV-SC: An Ensemble Learning-Based Model for Porcine Enterovirus Identification

## Dataset
NCBI: https://www.ncbi.nlm.nih.gov/
UniProt: https://www.uniprot.org/
VirusDIP: https://db.cngb.org/virusdip/
### Positive sample
<br><img width="788" height="259" alt="image" src="https://github.com/user-attachments/assets/a4ab2ad0-9f7a-4bcc-8c7a-7ab6a4729689" />
<br>
### Negative sample
<br><img width="770" height="136" alt="image" src="https://github.com/user-attachments/assets/300a047b-87ad-4f2b-b549-7adc0b0e1164" />
<br>
### Dataset structure diagram
<img width="2000" height="1529" alt="Figure3" src="https://github.com/user-attachments/assets/0698b402-25ec-4dd7-b57e-42c154ba206e" />

## Architecture diagram
<img width="1500" height="2000" alt="Figure2" src="https://github.com/user-attachments/assets/dc492ec3-faaa-4015-bccf-91afb8deb395" />

## Core Dependencies
| Library          | Version  |
|------------------|----------|
| numpy            | ≥1.20.0  |
| pandas           | ≥1.3.0   |
| scikit-learn     | ≥1.0.0   |
| tensorflow       | ≥2.6.0   |
| torch            | ≥1.10.0  |
| esm              | ≥0.5.0   |
| imbalanced-learn | ≥0.9.0   |

## Usage Instructions
Run the main script:
   ```bash
   python PEV-SC.py
