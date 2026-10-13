# LunarRM U-Net baseline

This baseline trains one frequency-conditioned U-Net for both challenge bands.
The three input channels are normalized terrain height, the transmitter one-hot
map, and a constant band channel (0 for 415 MHz and 1 for 5.775 GHz). The model
predicts normalized pathloss and is trained only on valid ray-traced pixels.

The official `lunar_dataset.py` in the dataset directory performs memory-mapped
loading, heightmap lookup, TX reconstruction, mask unpacking, normalization, and
aligned augmentation.

## Train

Check the dataset/model interface without training or changing weights:

```bash
cd /home/Users_Work_Space/zsfang/lunarRM
python check_setup.py
```

The committed full-training settings are in `run_train.sh`:

```bash
cd /home/Users_Work_Space/zsfang/lunarRM
bash run_train.sh
```

For a short CPU trial:

```bash
python train.py \
  --epochs 2 \
  --batch-size 4 \
  --base-channels 8 \
  --max-train-steps 100 \
  --max-val-batches 50 \
  --output-dir runs/cpu_trial
```

Metrics are appended to `metrics.jsonl`. `best.pt` is selected by the unweighted
mean of the 415 MHz and 5.775 GHz validation RMSE values in dB.

Checkpoints contain the model, optimizer, learning-rate scheduler, AMP scaler,
epoch, global step, and best metric. Resume with `--resume path/to/last.pt`.
