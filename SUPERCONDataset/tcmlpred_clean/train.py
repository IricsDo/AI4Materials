import torch
from torch.utils.data import DataLoader
import numpy as np
import config
from data import load_data
from model import EarlyStopping, TcMLPred


train_ds, test_ds, input_dim, y_std = load_data(
    "../archive/featurized.csv",
    config.TEST_SIZE,
    config.RANDOM_STATE
)


train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE
)
  
model = TcMLPred(input_dim)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=7,
    min_lr=1e-6
)

early_stopping = EarlyStopping(
    patience=20, 
    min_delta=0.05, 
    model_path=config.MODEL_PATH
)
reg_loss = torch.nn.MSELoss()
cls_loss = torch.nn.CrossEntropyLoss()


for epoch in range(config.EPOCHS):

    current_lr = optimizer.param_groups[0]['lr']
    
    model.train()

    total_loss = 0
    total_train_mse_scaled = 0
    
    for x, y_reg, y_cls in train_loader:

        tc_pred, cls_pred = model(x)


        loss1 = reg_loss(
            tc_pred.squeeze(),
            y_reg
        )

        loss2 = cls_loss(
            cls_pred,
            y_cls
        )
        
        precision_reg = torch.exp(-model.log_var_reg)
        loss_reg_weighted = precision_reg * loss1 + model.log_var_reg
        
        precision_cls = torch.exp(-model.log_var_cls)
        loss_cls_weighted = precision_cls * loss2 + model.log_var_cls
        
        loss = loss_reg_weighted + loss_cls_weighted

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_train_mse_scaled += loss1.item()
    
    train_rmse_kelvin = np.sqrt(total_train_mse_scaled / len(train_loader)) * y_std
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    total_val_mse_scaled = 0
    
    with torch.no_grad():
        for x_val, y_reg_val, y_cls_val in test_loader:
            tc_pred_val, _ = model(x_val)
            loss1_val = reg_loss(tc_pred_val.squeeze(), y_reg_val)
            total_val_mse_scaled += loss1_val.item()
            
    val_rmse_kelvin = np.sqrt(total_val_mse_scaled / len(test_loader)) * y_std
    
    print(f"Epoch {epoch:03d} | LR: {current_lr:.1e} | Loss: {avg_loss:>7.4f} | Temperature RMSE (train): {train_rmse_kelvin:>7.4f} K | Temperature RMSE (val): {val_rmse_kelvin:>7.4f} K")
    
    scheduler.step(val_rmse_kelvin)
    early_stopping(val_rmse_kelvin, model)
    
    if early_stopping.early_stop:
        print(f"📌 Đã kích hoạt Early Stopping tại Epoch {epoch}! Val RMSE tốt nhất: {early_stopping.best_loss:.4f} Kelvin.")
        break
    
torch.save(
    model.state_dict(),
    config.MODEL_PATH
)