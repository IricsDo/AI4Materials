Requirements: pyTorch 

To run the code, the following procedure is set up:
Generate a test/train split by running: python makemodel.py (makemodel scrambles the two sets of "fail" and "success").

This will produce two model files: train_ds and val_ds (for training and validation sets, respectively).

Then run: python train_modular_net.py for CNN
 OR
run train_modular_net_FC.py for FC
