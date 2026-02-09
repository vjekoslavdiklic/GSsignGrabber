# GSsignGrabber

Grab a Google Street View image, detect traffic signs with YOLO, crop them, and classify each sign with a second neural network.

## What this does
- Downloads a Street View image (or uses a local file).
- Detects traffic signs with YOLO.
- Crops each detection.
- Classifies each crop with a separate classifier.
- Writes crops + a CSV log.

## Setup
Install dependencies (examples):

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision pillow requests
```

## Usage
### Option A: Local image
```bash
python src/gs_sign_grabber.py \
  --image-path /path/to/street_view.jpg \
  --yolo-weights /path/to/yolo-weights.pt \
  --classifier-weights /path/to/classifier.pt \
  --classifier-labels /path/to/labels.txt
```

### Option B: Google Street View
```bash
python src/gs_sign_grabber.py \
  --lat 37.4219999 \
  --lon -122.0840575 \
  --api-key "$GOOGLE_API_KEY" \
  --yolo-weights /path/to/yolo-weights.pt \
  --classifier-weights /path/to/classifier.pt \
  --classifier-labels /path/to/labels.txt
```

### Optional arguments
- `--yolo-classes "stop sign,speed limit"` to keep only specific classes by name.
- `--output-dir outputs` for the crops and CSV.

The script writes:
- `outputs/crops/sign_XXX.jpg`
- `outputs/sign_log.csv`

## Notes
- You need to provide YOLO weights trained on traffic sign detection.
- You need a classifier checkpoint and a labels file (one class name per line).
- Google Street View requires an API key and a billing-enabled project.
