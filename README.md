## This repository contains the code for designing GFP variants with high brightness and thermal stability.

## Repository Structure
- `run_design.py` : Main design pipeline
- `submission_2026.csv` : Final 6 submitted sequences
- `requirements.txt` : Python dependencies

## Requirements
- Python 3.8+
- macOS / Linux / Windows

Install dependencies:
```bash
pip install -r requirements.txt
```
Data Files Required

Place these official files in the same folder as run_design.py:

- GFP_data.xlsx
- AAseqs_of_4_GFP_proteins.txt
- Exclusion_List.csv

## How to Run
```bash
python3 run_design.py
```
After running, you will get submission_2026.csv containing the top 6 candidate sequences.

## Method Summary
- Backbone: sfGFP (wild-type sequence)
- Feature extraction: ESM-2 embeddings (esm2_t30_150M_UR50D)
- Predictor: Random Forest Regressor
- Design strategy: Random mutagenesis (5–20 mutations per sequence) while protecting key residues (chromophore and known stabilizing sites)
- Filtering: Exclude known sequences from Exclusion_List.csv; limit proline count; select top 6 by predicted brightness

## Team Name
乱折一通也能赢
