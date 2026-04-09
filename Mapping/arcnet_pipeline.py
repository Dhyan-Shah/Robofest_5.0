import cv2
import glob
import numpy as np
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH      = "C:\\Users\\Dell\\Desktop\\Robofest 5.0\\Codes\\Models\\Mine_Detection\\best.pt"          # your trained YOLO model
FRAMES_FOLDER   = "C:\\Users\\Dell\\Desktop\\frames\\*.jpeg"      # folder with RPi camera frames
ORTHO_PATH      = "orthomosaic.jpg"   # stitched output (auto saved)
OUTPUT_PATH     = "mine_overlay.jpg"  # final output with detections

FIELD_WIDTH_M   = 20.0   # real world field width in meters
CONF_THRESHOLD  = 0.45   # detection confidence threshold
EXCL_RADIUS_M   = 1.2    # exclusion zone radius in meters
BOX_HALF        = 22     # detection box half-size in pixels

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — STITCH FRAMES INTO ORTHOMOSAIC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[ STEP 1 ] Loading frames...")
paths = sorted(glob.glob(FRAMES_FOLDER))

if len(paths) == 0:
    print("ERROR: No frames found in 'frames/' folder!")
    exit()

print(f"          Found {len(paths)} frames")
images = [cv2.imread(p) for p in paths]
images = [img for img in images if img is not None]  # remove failed reads

print(f"[ STEP 1 ] Stitching {len(images)} images — this may take 30–60 seconds...")
stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
status, stitched = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    cv2.imwrite(ORTHO_PATH, stitched)
    print(f"[ STEP 1 ] ✓ Orthomosaic saved → {ORTHO_PATH}")
else:
    error_codes = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "Need more images (add more frames with overlap)",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography failed (increase frame overlap)",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera params failed (check image quality)",
    }
    msg = error_codes.get(status, f"Unknown error code {status}")
    print(f"[ STEP 1 ] ✗ Stitching failed: {msg}")
    print("           TIP: Make sure frames have 60-70% overlap and consistent altitude")
    exit()

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — DETECT MINES ON STITCHED IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[ STEP 2 ] Loading orthomosaic...")
img = cv2.imread(ORTHO_PATH)
H, W = img.shape[:2]
print(f"           Image size: {W}x{H} px")

PX_PER_M = W / FIELD_WIDTH_M
EXCL_R   = int(EXCL_RADIUS_M * PX_PER_M)

print(f"[ STEP 2 ] Running YOLO detection (conf={CONF_THRESHOLD})...")
model   = YOLO(MODEL_PATH)

# Tiled detection for better accuracy on large stitched image
TILE_SIZE = 640
OVERLAP   = 100
mines_px  = []  # pixel coords
mines_m   = []  # real world coords in meters

for y in range(0, H, TILE_SIZE - OVERLAP):
    for x in range(0, W, TILE_SIZE - OVERLAP):
        tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
        if tile.shape[0] < 100 or tile.shape[1] < 100:
            continue

        results = model(tile, conf=CONF_THRESHOLD, verbose=False)[0]

        for box in results.boxes:
            cx, cy = box.xywh[0][:2].tolist()
            # Convert tile coords → full image coords
            full_cx = cx + x
            full_cy = cy + y
            mines_px.append((int(full_cx), int(full_cy)))

# Remove duplicate detections from overlapping tiles
def deduplicate(mines, min_dist=30):
    kept = []
    for m in mines:
        too_close = False
        for k in kept:
            dist = np.sqrt((m[0]-k[0])**2 + (m[1]-k[1])**2)
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(m)
    return kept

mines_px = deduplicate(mines_px)

# Convert pixel coords → real world meters
for (px, py) in mines_px:
    mx = (px / W) * FIELD_WIDTH_M
    my = (1 - py / H) * 100.0   # flip Y, scale to 100m field length
    mines_m.append((round(mx, 2), round(my, 2)))

print(f"[ STEP 2 ] ✓ Detected {len(mines_px)} mines")
for i, (mx, my) in enumerate(mines_m):
    print(f"           Mine {i+1:02d}: X={mx}m, Y={my}m")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — DRAW OVERLAY ON IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[ STEP 3 ] Drawing overlay...")

# Exclusion zone filled halos (semi-transparent)
overlay = img.copy()
for (mx, my) in mines_px:
    cv2.circle(overlay, (mx, my), EXCL_R, (0, 40, 220), -1)
cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

# Exclusion zone rings
for (mx, my) in mines_px:
    cv2.circle(img, (mx, my), EXCL_R, (0, 80, 255), 2)

# Detection boxes + center dots
for (mx, my) in mines_px:
    cv2.rectangle(img,
        (mx - BOX_HALF, my - BOX_HALF),
        (mx + BOX_HALF, my + BOX_HALF),
        (0, 0, 255), 2)
    cv2.circle(img, (mx, my), 4, (0, 0, 255), -1)
    cv2.circle(img, (mx, my), 4, (255, 255, 255), 1)

# ── Save final output ─────────────────────────────────────────────────────────
cv2.imwrite(OUTPUT_PATH, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"[ STEP 3 ] ✓ Final overlay saved → {OUTPUT_PATH}")

# ── Save mine coordinates for path algorithm ──────────────────────────────────
import json
with open("mines.json", "w") as f:
    json.dump({"mines_meters": mines_m, "total": len(mines_m)}, f, indent=2)
print(f"[ STEP 3 ] ✓ Mine coordinates saved → mines.json")

print("\n═══════════════════════════════════════════")
print(f"  PIPELINE COMPLETE")
print(f"  Mines detected : {len(mines_px)}")
print(f"  Output image   : {OUTPUT_PATH}")
print(f"  Coordinates    : mines.json")
print(f"  Next step      : feed mines.json into hybrid.py")
print("═══════════════════════════════════════════\n")