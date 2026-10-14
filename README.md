# BN4601A Individual Assignment: Breast Cancer Classification on Mammograms

Individual assignment for BN4601A Medical Artificial Intelligence (NUS, AY2026/27).

A YOLO11s classification model trained on CBIS-DDSM mass cases to label a whole
screening mammogram as **benign** or **malignant**, deployed as a Gradio app.

Held-out test performance (361 images, patient-disjoint from training):

| Threshold on P(malignant) | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| 0.50 | 0.767 | 0.701 | 0.690 | 0.708 |
| 0.24 (referral, chosen on validation for ≥85% sensitivity) | 0.767 | 0.645 | 0.786 | 0.551 |

**Educational demonstration only. Not a medical device and not for diagnosis.**

## Contents

| Path | What it is |
|---|---|
| `Assignment.ipynb` | Main notebook: data mapping, patient-level split, training, evaluation |
| `AppShowcase.ipynb` | Demo of the app |
| `create_notebook.py` | Script that generated the assignment notebook |
| `app.py`, `best.pt`, `example_*.jpg` | Gradio app and the deployed classifier |
| `space_deploy/` | Files as deployed to the Hugging Face Space |
| `runs/` | Ultralytics training runs (plots, metrics, `best.pt`) |
| `detection_trial/` | Side experiment: mass detection with bounding boxes vs. the classifier |
| `report_assets/`, `video_demo_images/` | Figures for the report and video |
| `csv/` | CBIS-DDSM metadata CSVs used by the notebooks |
| `environment.yml`, `requirements.txt` | Conda environment (`bn4601`) and app requirements |

## Re-running

The images are not in this repo. Download the CBIS-DDSM JPEG dataset from Kaggle
and place its `jpeg/` folder at the repo root (next to `csv/`). The notebooks
rebuild `cls_dataset/` and `detection_trial/det_dataset/` from it.

```
conda env create -f environment.yml
conda activate bn4601
python app.py
```

## Data

CBIS-DDSM (Curated Breast Imaging Subset of DDSM), The Cancer Imaging Archive,
licensed CC BY 3.0. Lee, R. S. et al. (2017). A curated mammography data set for
use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
