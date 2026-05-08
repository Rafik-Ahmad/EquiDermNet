# EquiDermNet

**Equitable Skin Lesion Classification via Evidential Deep Learning and Adversarial Fairness**

EquiDermNet is a fairness-aware deep learning framework for dermoscopic image classification. It combines **Evidential Deep Learning (EDL)** for uncertainty-calibrated predictions with **adversarial debiasing** and **orthogonal feature disentanglement** to produce skin tone-equitable diagnostic outputs.

---

## 🧠 Key Features

- **DenseNet-121 backbone** with dual projection heads — one for lesion features (`z_l`), one for skin-tone features (`z_s`)
- **Evidential Deep Learning (EDL)** loss with annealed KL regularization for uncertainty estimation
- **Gradient Reversal Layer (GRL)** for adversarial skin tone debiasing
- **Orthogonality loss** to disentangle lesion and skin-tone latent spaces
- **Weighted random sampling** to handle severe class imbalance in HAM10000
- **Fairness metrics** (EOD — Equalized Odds Difference) tracked per epoch

---

## 📁 Project Structure

```
EquiDermNet/
├── src/
│   ├── config.py        # Paths and hyperparameters
│   ├── model.py         # EquiDermNet architecture (DenseNet-121 + GRL + dual heads)
│   ├── dataloader.py    # HAM10000 dataset loader with weighted sampling
│   ├── train.py         # Main training loop with fairness-aware losses
│   └── utils.py         # EDL loss, orthogonality loss, metrics logger
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🗃️ Dataset

This project uses the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

**Classes (7):**

| Code | Condition |
|------|-----------|
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `bcc` | Basal cell carcinoma |
| `akiec` | Actinic keratoses |
| `bkl` | Benign keratosis |
| `df` | Dermatofibroma |
| `vasc` | Vascular lesions |

**Skin tone labeling** uses the ITA (Individual Typology Angle) score:
- `ITA > 28` → Light skin (label `0`)
- `ITA ≤ 28` → Dark skin (label `1`)

### Directory layout expected by the code

```
equidermnet_local/
└── data/
    ├── images/
    │   ├── ISIC_0024306.jpg
    │   └── ...
    └── HAM10000_metadata.csv
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/EquiDermNet.git
cd EquiDermNet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths

Edit `src/config.py` to point `DRIVE_ROOT` and `LOCAL_ROOT` to your environment:

```python
DRIVE_ROOT = '/content/drive/MyDrive/EquiDermNet_Project'   # Google Drive (Colab)
LOCAL_ROOT  = '/content/equidermnet_local'                   # Local data path
```

---

## 🚀 Training

```bash
cd src
python train.py
```

Training logs AUC and EOD every epoch. The best model checkpoint is saved to `CHECKPOINT_DIR/best_model.pth`. Final metrics are written to `results_equiderm.csv` (fairness-aware run) or `results_baseline.csv` (baseline, `LAMBDA_FAIR=0`).

---

## 📐 Loss Function

The total training loss is:

```
L = L_EDL + λ_fair · L_adv + λ_ortho · L_ortho
```

| Term | Description |
|------|-------------|
| `L_EDL` | Evidential Bayes risk + annealed KL divergence (Dirichlet prior) |
| `L_adv` | Binary cross-entropy adversarial loss on skin tone prediction (with GRL) |
| `L_ortho` | Frobenius norm of cross-correlation between `z_l` and `z_s` |

Default: `λ_fair = 0.1`, `λ_ortho = 0.1`

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| `Global_AUC` | One-vs-rest macro AUC across all 7 classes |
| `Global_ACC` | Overall classification accuracy |
| `Light_ACC / Dark_ACC` | Per-group accuracy |
| `Light_Sens / Dark_Sens` | Per-group mean sensitivity |
| `EOD` | Equalized Odds Difference — `|Light_Sens − Dark_Sens|` |

---

## 🔧 Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BATCH_SIZE` | 32 | |
| `LR` | 1e-4 | Adam optimizer |
| `EPOCHS` | 30 | |
| `LAMBDA_FAIR` | 0.1 | Reduced from 1.0 to prevent feature collapse |
| `LAMBDA_ORTHO` | 0.1 | |

---

## 📦 Requirements

See `requirements.txt`. Core dependencies:

- `torch >= 1.13`
- `torchvision`
- `scikit-learn`
- `pandas`
- `numpy`
- `Pillow`

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@misc{equidermnet2025,
  title   = {EquiDermNet: Equitable Skin Lesion Classification via Evidential Deep Learning and Adversarial Fairness},
  author  = {<Your Name>},
  year    = {2025},
  url     = {https://github.com/<your-username>/EquiDermNet}
}
```

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
