# ====================================================
# Library
# ====================================================
import os
import time
from torch.cuda import amp
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score, log_loss
from PIL import ImageFile
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import CFG
import Utils
import Dataset
import Model
import Engine

train_df = pd.read_csv('train_df')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VERSION = 1

# ====================================================
# Directory settings
# ====================================================
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ====================================================
# Train loop
# ====================================================

def train_loop(folds, fold):
    scaler = amp.GradScaler()
    Utils.LOGGER.info(f"========== fold: {fold} training ==========")
    
    if CFG.debug:
        train_folds = folds[folds['fold'] != fold].sample(50)
        valid_folds = folds[folds['fold'] == fold].sample(50)
        
    else:
        train_folds = folds[folds['fold'] != fold]
        valid_folds = folds[folds['fold'] == fold]
        
        
    train_images_path = train_folds.path_jpeg.values
    train_targets = train_folds.label.values
        
    valid_images_path = valid_folds.path_jpeg.values
    valid_targets = valid_folds.label.values
        
    train_dataset = Dataset.classificationDataset(dataframe=train_folds,
                                image_paths=train_images_path,
                                targets=train_targets,
                                augmentations=Dataset.get_transforms(data='train'))
        
    valid_dataset = Dataset.classificationDataset(dataframe=valid_folds,
                                image_paths=valid_images_path,
                                targets=valid_targets,
                                augmentations=Dataset.get_transforms(data='valid'))
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.batch_size, shuffle=True
    )
        
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.batch_size, shuffle=False
    )
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, **CFG.reduce_params)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, **CFG.cosanneal_params)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, **CFG.cosanneal_res_params)
        return scheduler
    
    # ====================================================
    # model & optimizer
    # ====================================================
        
    model = Model.Effnet_Landmark(CFG.model_name, CFG.target_size)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_loss = np.inf
    best_acc = -np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = Engine.train_fn(fold, train_loader, model, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = Engine.valid_fn(valid_loader, model, criterion, device)
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        
        preds_label = np.argmax(preds, axis=1)
        Accuracy = accuracy_score(preds_label, valid_targets)
        score = Utils.get_score(valid_targets, preds)
        
        elapsed = time.time() - start_time

        Utils.LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        Utils.LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        Utils.LOGGER.info(f'Epoch {epoch+1} - Accuracy: {Accuracy:.4f}')

            
        if Accuracy > best_acc:
            best_acc = Accuracy
            Utils.LOGGER.info(f'Epoch {epoch+1} - Save Best Accuracy: {best_acc:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds_loss': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_Accuracy.pth')
            

        
        
    valid_folds[CFG.preds_col] = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_Accuracy.pth', 
                                      map_location=torch.device('cpu'))['preds_loss']

    return valid_folds

# ====================================================
# main
# ====================================================
def main():

    """
    Prepare: 1.train 
    """

    def get_result(result_df):
        preds_loss = result_df[CFG.preds_col].values
        labels = result_df["label"].values
        preds = np.argmax(preds_loss, axis=1)
        Accuracy = accuracy_score(labels, preds)
        Utils.LOGGER.info(f'Accuracy with best weights: {Accuracy:<.4f}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.nfolds):
            if fold in CFG.trn_folds:
                _oof_df = train_loop(train_df, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                Utils.LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        Utils.LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df[['image_id','bacterial_leaf_blight', 'bacterial_leaf_streak',
       'bacterial_panicle_blight', 'blast', 'brown_spot', 'dead_heart',
       'downy_mildew', 'hispa', 'normal', 'tungro','label']].to_csv(OUTPUT_DIR+f'{CFG.model_name}_oof_rgb_df_version{VERSION}.csv', index=False)

if __name__ == "__main__":
    main()