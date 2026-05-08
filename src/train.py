import torch
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F

from config import *
from model import EquiDermNet
from dataloader import get_loaders
from utils import edl_loss, orthogonality_loss, ResearchMetricsLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"🚀 Starting EquiDermNet Training on {device}")
    
    metadata = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
    train_loader, val_loader = get_loaders(BATCH_SIZE, DATA_DIR, metadata)
    
    model = EquiDermNet(num_classes=NUM_CLASSES).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Removed 'verbose=True' because it causes a crash in new PyTorch versions.
    # We will manually print the LR change below.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3
    )
    
    logger = ResearchMetricsLogger(NUM_CLASSES)
    history = []
    
    best_auc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for imgs, labels, skins in train_loader:
            imgs, labels, skins = imgs.to(device), labels.to(device), skins.to(device).float()
            
            optimizer.zero_grad()
            logits, skin_pred, z_l, z_s = model(imgs)
            
            # Losses
            l_edl = edl_loss(logits, labels, epoch, EPOCHS, NUM_CLASSES)
            l_adv = F.binary_cross_entropy_with_logits(skin_pred, skins.unsqueeze(1))
            l_ortho = orthogonality_loss(z_l, z_s)
            
            loss = l_edl + LAMBDA_FAIR * l_adv + LAMBDA_ORTHO * l_ortho
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        logger.reset()
        with torch.no_grad():
            for imgs, labels, skins in val_loader:
                imgs, labels, skins = imgs.to(device), labels.to(device), skins.to(device)
                logits, _, _, _ = model(imgs)
                logger.update(logits, labels, skins)
                
        metrics = logger.get_report()
        metrics['Epoch'] = epoch + 1
        metrics['Train_Loss'] = total_loss / len(train_loader)
        metrics['LR'] = optimizer.param_groups[0]['lr']
        history.append(metrics)
        
        print(f"Epoch {epoch+1} | AUC: {metrics['Global_AUC']:.3f} | EOD: {metrics['EOD']:.3f} | LR: {metrics['LR']:.1e}")
        
        # Step the scheduler based on AUC
        scheduler.step(metrics['Global_AUC'])
        
        # Manually check if LR changed to print it
        if epoch > 0 and metrics['LR'] < history[-2]['LR']:
            print(f"  📉 Learning Rate reduced to {metrics['LR']:.1e}!")
        
        # Save Best Model Only
        if metrics['Global_AUC'] > best_auc:
            best_auc = metrics['Global_AUC']
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print("  --> New Best Model Saved!")

    # Save Results
    df = pd.DataFrame(history)
    csv_name = 'results_equiderm.csv' if LAMBDA_FAIR > 0 else 'results_baseline.csv'
    df.to_csv(os.path.join(LOG_DIR, csv_name), index=False)
    print(f"✅ Training Complete. Best AUC: {best_auc:.4f}")

if __name__ == '__main__':
    main()
