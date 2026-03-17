import torch
from torch.utils.data import DataLoader

import config
from data import load_data
from model import TcMLPred


train_ds, test_ds, input_dim = load_data(
    "feature.csv",
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

reg_loss = torch.nn.MSELoss()
cls_loss = torch.nn.CrossEntropyLoss()


for epoch in range(config.EPOCHS):

    model.train()

    total_loss = 0

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

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} loss {total_loss}")

torch.save(
    model.state_dict(),
    config.MODEL_PATH
)