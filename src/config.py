import os

# --- PATHS ---
DRIVE_ROOT = '/content/drive/MyDrive/EquiDermNet_Project'
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, 'checkpoints')
LOG_DIR = os.path.join(DRIVE_ROOT, 'results_csv')
FIG_DIR = os.path.join(DRIVE_ROOT, 'paper_figures')

LOCAL_ROOT = '/content/equidermnet_local'
DATA_DIR = os.path.join(LOCAL_ROOT, 'data')

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30
NUM_CLASSES = 7

# --- MODIFIED PARAMETERS ---
# Previous 1.0 was too aggressive (caused collapse)
# 0.1 allows the model to learn features while still punishing bias
LAMBDA_FAIR = 0.1
LAMBDA_ORTHO = 0.1
