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
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE,
    num_workers=4,
    pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TcMLPred(input_dim).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LR,
    weight_decay=1e-4
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

        x = x.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)
        
        tc_pred, cls_pred = model(x)

        loss1 = reg_loss(
            tc_pred.squeeze(),
            y_reg
        )

        loss2 = cls_loss(
            cls_pred,
            y_cls
        )
        
        # precision_reg = torch.exp(-model.log_var_reg)
        # loss_reg_weighted = precision_reg * loss1 + model.log_var_reg
        
        # precision_cls = torch.exp(-model.log_var_cls)
        # loss_cls_weighted = precision_cls * loss2 + model.log_var_cls
        
        # loss = loss_reg_weighted + loss_cls_weighted
        loss = model.multitask_loss(tc_pred, cls_pred, y_reg, y_cls)
    
        optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()

        total_loss += loss.item()
        total_train_mse_scaled += loss1.item()


    print("reg weight:", torch.exp(-model.log_var_reg).item(),
    "cls weight:", torch.exp(-model.log_var_cls).item())
    
    train_rmse_kelvin = np.sqrt(total_train_mse_scaled / len(train_ds)) * y_std
    avg_loss = total_loss / len(train_loader)
    
    model.eval()
    total_val_mse_scaled = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_val, y_reg_val, y_cls_val in test_loader:

            x_val = x_val.to(device)
            y_reg_val = y_reg_val.to(device)
            y_cls_val = y_cls_val.to(device)

            tc_pred_val, cls_pred_val = model(x_val)

            predicted = torch.argmax(cls_pred_val, dim=1)

            correct += (predicted == y_cls_val).sum().item()
            total += y_cls_val.size(0)

            loss1_val = reg_loss(tc_pred_val.squeeze(-1), y_reg_val)
            total_val_mse_scaled += loss1_val.item()
            
    val_rmse_kelvin = np.sqrt(total_val_mse_scaled / len(test_ds)) * y_std
    val_acc = correct / total
    
    print(
    f"Epoch {epoch:03d} | LR: {current_lr:.1e} | "
    f"Loss: {avg_loss:>7.4f} | "
    f"Train RMSE: {train_rmse_kelvin:>7.4f} K | "
    f"Val RMSE: {val_rmse_kelvin:>7.4f} K | "
    f"Val Acc: {val_acc:.3f}"
    )    
    scheduler.step(val_rmse_kelvin)
    early_stopping(val_rmse_kelvin, model)
    
    if early_stopping.early_stop:
        print(f"📌 Đã kích hoạt Early Stopping tại Epoch {epoch}! Val RMSE tốt nhất: {early_stopping.best_loss:.4f} Kelvin.")
        break
    
torch.save(
    model.state_dict(),
    config.MODEL_PATH
)