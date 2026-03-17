import torch
from torch.utils.data import DataLoader

import config
from data import load_data
from model import TcMLPred
from utils import classification_accuracy


train_ds, test_ds, input_dim = load_data(
    "feature.csv",
    config.TEST_SIZE,
    config.RANDOM_STATE
)

test_loader = DataLoader(
    test_ds,
    batch_size=config.BATCH_SIZE
)

model = TcMLPred(input_dim)

model.load_state_dict(
    torch.load(config.MODEL_PATH)
)

model.eval()

reg_loss = torch.nn.MSELoss()

total_loss = 0
acc = 0

with torch.no_grad():

    for x, y_reg, y_cls in test_loader:

        tc_pred, cls_pred = model(x)

        total_loss += reg_loss(
            tc_pred.squeeze(),
            y_reg
        ).item()

        acc += classification_accuracy(
            cls_pred,
            y_cls
        )

print("Regression MSE:", total_loss)
print("Classification accuracy:", acc / len(test_loader))