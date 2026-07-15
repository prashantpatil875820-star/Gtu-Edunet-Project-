# 🔥 Forest Fire Risk Prediction (CNN + Live Satellite Imagery)

Deep-learning pipeline that classifies forest-fire risk from satellite
imagery, and can pull **live, up-to-date imagery** for any location on
demand.

## How it works

| Component | Role | Data source |
|---|---|---|
| `fetch_live_satellite.py` | Pulls a live true-color satellite image + active-fire hotspots for a lat/lon | NASA GIBS (imagery, no key needed) + NASA FIRMS (active fires, free key) |
| `model.py` | CNN architecture (from-scratch + optional MobileNetV2 transfer learning) | — |
| `train.py` | Trains the CNN on a labeled image dataset | Your dataset (see below) |
| `predict_live.py` | Fetches a live image and classifies it, cross-checked against FIRMS | — |

## 1. Setup

```bash
pip install -r requirements.txt
```

Get a **free** NASA FIRMS key (~1 min, no cost, no card):
https://firms.modaps.eosdis.nasa.gov/api/map_key/

Create a `.env` file in this folder:
```
FIRMS_MAP_KEY=your_key_here
```

NASA GIBS (the actual satellite imagery) needs **no key** — it's a public
WMS service.

## 2. Get training data

You need labeled images in this structure:

```
data/dataset/train/no_fire_risk/*.jpg
data/dataset/train/fire_risk/*.jpg
data/dataset/val/no_fire_risk/*.jpg
data/dataset/val/fire_risk/*.jpg
```

Two ways to build this:
- **Fastest:** download a labeled dataset like Kaggle's "Wildfire
  Prediction Dataset (Satellite Images)" (Sentinel-2, already
  labeled fire/nofire) and drop it into the folders above.
- **Fully live:** loop `fetch_live_satellite.py` over known fire-prone
  and safe coordinates, and use the FIRMS `hotspot_count` as a weak
  auto-label (hotspots present → `fire_risk`, none → `no_fire_risk`).

## 3. Train

```bash
python train.py
```

This trains a 4-block CNN (Conv→BN→ReLU→Pool, GlobalAveragePooling head)
with built-in augmentation, saves the best checkpoint by validation AUC,
and prints final metrics. For datasets over a few thousand images, switch
to transfer learning by editing the last line of `train.py`:

```python
main(use_transfer_learning=True)   # uses MobileNetV2 backbone
```

## 4. Predict on a LIVE satellite image

```bash
python predict_live.py --lat 34.2 --lon -118.1
```

This will:
1. Pull yesterday's true-color satellite tile centered on that point from NASA GIBS
2. Run it through your trained CNN
3. Report predicted risk + confidence
4. Cross-check against real active-fire detections from FIRMS for a sanity check

## Notes & next steps
- `IMAGE_SIZE`, class list, and multiclass vs binary mode are all in `config.py`.
- Swap in Sentinel-2 (via Sentinel Hub) or Landsat imagery for higher resolution
  than GIBS if you need finer-grained detection — same interface, just replace
  `fetch_gibs_image`.
- For production use, add temporal features (e.g., a rolling window of NDVI /
  drought index) alongside the image — single-image CNNs miss fuel-dryness trends
  that matter a lot for real fire risk.
