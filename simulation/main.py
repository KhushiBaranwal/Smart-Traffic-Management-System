import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO


# ── Models ─────────────────────────────────────────
custom_model = YOLO(r"C:\Users\Khushi Baranwal\runs\detect\train11\weights\best.pt")
base_model   = YOLO("yolov8n.pt") 

vehicle_cap = cv2.VideoCapture("TRAFFIC_DATA_NEW/lane4.mp4") #lane1
ped_cap     = cv2.VideoCapture("TRAFFIC_DATA_NEW/lane2.mp4")   #lane2

if not vehicle_cap.isOpened():
    print("[ERROR] lane5.mp4 not found"); exit()
if not ped_cap.isOpened():
    print("[ERROR] dis.mp4 not found"); exit()

# ── Class IDs (custom model) ────────────────────────
EMERGENCY_IDS  = {5, 6, 7}
VEHICLE_IDS    = {8, 9, 10, 11, 12,13, 14, 17, 18, 19, 20, 21, 22}
PEDESTRIAN_IDS = {15}
DISABLED_IDS   = {1, 4}
ALL_RELEVANT   = EMERGENCY_IDS | VEHICLE_IDS | PEDESTRIAN_IDS | DISABLED_IDS

BASE_PERSON_ID   = 0
BASE_VEHICLE_IDS = {2, 3, 5, 7}

# ── [D1] Lower conf threshold (was 0.30 / 0.40) ────
CONF_CUSTOM  = 0.25   # fix: catch more detections
CONF_BASE    = 0.30
IOU_THRESH   = 0.45   # to avoid overlapping of two models
IMGSZ        = 640

SNAP_FRAMES  = 5
SNAP_NEEDED  = 3

# ── [L4] Starvation: max wait cap (seconds) ────────
MAX_WAIT_CAP = 45   # force-switch if lane waits > this long

# ── Layout ─────────────────────────────────────────
FEED_W  = 560
FEED_H  = 400
GAP     = 24
PANEL_W = 150
HEADER  = 72
FOOTER  = 130      # extra row for wait-time display

TOTAL_W = PANEL_W + FEED_W + GAP + FEED_W + PANEL_W
TOTAL_H = HEADER + FEED_H + FOOTER

# ── Colors (BGR) ───────────────────────────────────
BG        = (18,  18,  24)
DARK      = (10,  10,  15)
PANEL_BG  = (26,  26,  36)
ACCENT    = (0,  210, 255)
GREEN_ON  = (30, 220,  80)
RED_ON    = (40,  40, 220)
ORANGE_ON = (0,  140, 255)
DIM_BULB  = (45,  45,  58)
WHITE     = (235, 235, 245)
GRAY      = (120, 120, 140)
DIVIDER   = (40,  40,  55)
PURPLE    = (180,  60, 180)

# ── [D4] ROI Polygon (road area only, ignore sky) ── 30% height
# These are default fractions — auto-applied to FEED_W x FEED_H frames
# Adjust per camera angle if needed
def make_roi_mask(h, w):
    """Bottom 70% trapezoid — ignores sky and overhead"""
    pts = np.array([
        [0,          int(h*0.30)],
        [w,          int(h*0.30)],
        [w,          h],
        [0,          h],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

ROI_MASK = make_roi_mask(FEED_H, FEED_W)

# ── [D2/L3] Night-mode detection + preprocessing ──
def is_low_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < 60   # threshold for night/dark scene

_sharpen_k = np.array([[0,   -0.5,  0  ],
                        [-0.5, 3.0, -0.5],
                        [0,   -0.5,  0  ]], dtype=np.float32)

def preprocess(frame):
    # Always: CLAHE  -> Contrast Limited Adaptive Histogram Equalization
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # [D2] Night: extra gamma boost
    if is_low_light(frame):
        gamma = 1.8
        lut = np.array([((i/255.0)**( 1.0/gamma))*255
                        for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, lut)

    frame = cv2.filter2D(frame, -1, _sharpen_k)
    return frame

# ── [D4] Apply ROI mask to frame ───────────────────
def apply_roi(frame):
    out = frame.copy()
    out[ROI_MASK == 0] = 0
    return out

# ── [M1] Optical-flow motion tracker ───────────────
class MotionTracker:
    """Tracks which detected boxes contain moving objects."""
    def __init__(self):
        self.prev_gray = None

    def moving_boxes(self, frame, boxes):
        """Returns set of indices from boxes that are MOVING."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        moving = set()

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            for i, (x1,y1,x2,y2) in enumerate(boxes):
                roi = thresh[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
                if roi.size > 0 and np.mean(roi) > 8:
                    moving.add(i)
        else:
            # First frame: mark all as moving (assume active)
            moving = set(range(len(boxes)))

        self.prev_gray = gray.copy()
        return moving

motion1 = MotionTracker()
motion2 = MotionTracker()

# ── Label helper ───────────────────────────────────
def _label(img, text, x, y, color):
    y0 = max(y - 6, 14)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.rectangle(img, (x, y0-th-4), (x+tw+6, y0+3), (0,0,0), -1)
    cv2.putText(img, text, (x+3, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

# ── [M4] Pedestrian speed estimator ────────────────
class PedSpeedTracker:
    """Tracks pedestrian box centers across frames; flags slow walkers."""
    def __init__(self, history=8):
        self.history = deque(maxlen=history)

    def update(self, boxes_centers):
        self.history.append(boxes_centers)

    def is_slow(self):
        """Returns True if average movement per frame is low."""
        if len(self.history) < 3:
            return False
        movements = []
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]
            if prev and curr:
                # match closest centers
                for cx, cy in curr:
                    dists = [np.hypot(cx-px, cy-py) for px,py in prev]
                    movements.append(min(dists) if dists else 0)
        avg = np.mean(movements) if movements else 0
        return avg < 4.0   # px/frame threshold for "slow walker"

ped_speed_tracker = PedSpeedTracker()

# ── Core detection ─────────────────────────────────
def detect(frame, side, draw=True, motion_tracker=None):
    """
    Returns: (annotated_frame, count, has_disabled, has_emergency,
               stationary_count, slow_ped)
    """
    proc = preprocess(apply_roi(frame.copy()))
    out  = frame.copy() if draw else frame

    count      = 0
    disabled   = False
    emergency  = False
    stationary = 0
    slow_ped   = False
    all_boxes  = []
    box_types  = []   # 'emergency','disabled','vehicle','person'

    # ── 1. Custom model ──
    c_results = custom_model(proc, conf=CONF_CUSTOM,
                             iou=IOU_THRESH, imgsz=IMGSZ,
                             verbose=False, augment=False)
    drawn_boxes = []

    for r in c_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id not in ALL_RELEVANT:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = custom_model.names[cls_id]

            if cls_id in EMERGENCY_IDS:
                emergency = True; count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('emergency')
                if draw:
                    cv2.rectangle(out,(x1,y1),(x2,y2),(0,140,255),3)
                    _label(out, f"EMRG:{label} {conf:.2f}", x1, y1, (0,140,255))
                drawn_boxes.append((x1,y1,x2,y2))

            elif cls_id in DISABLED_IDS:
                disabled = True; count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('disabled')
                if draw:
                    cv2.rectangle(out,(x1,y1),(x2,y2),(40,40,230),2)
                    _label(out, f"DISABLED:{label} {conf:.2f}", x1, y1, (40,40,230))
                drawn_boxes.append((x1,y1,x2,y2))

            elif cls_id in VEHICLE_IDS and side == "vehicle":
                count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('vehicle')
                if draw:
                    cv2.rectangle(out,(x1,y1),(x2,y2),(0,210,255),2)
                    _label(out, f"{label} {conf:.2f}", x1, y1, (0,210,255))
                drawn_boxes.append((x1,y1,x2,y2))

            elif cls_id in PEDESTRIAN_IDS and side == "pedestrian":
                count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('person')
                if draw:
                    cv2.rectangle(out,(x1,y1),(x2,y2),(60,220,80),2)
                    _label(out, f"person {conf:.2f}", x1, y1, (60,220,80))
                drawn_boxes.append((x1,y1,x2,y2))

    # ── 2. Base model fallback ──
    b_results = base_model(proc, conf=CONF_BASE,
                           iou=IOU_THRESH, imgsz=IMGSZ,
                           verbose=False, augment=False)
    for r in b_results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if _overlaps(x1,y1,x2,y2, drawn_boxes):
                continue

            if cls_id == BASE_PERSON_ID and side == "pedestrian":
                count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('person')
                if draw:
                    cv2.rectangle(out,(x1,y1),(x2,y2),(60,220,80),2)
                    _label(out, f"person(b) {conf:.2f}", x1, y1, (60,220,80))
                drawn_boxes.append((x1,y1,x2,y2))

            elif cls_id in BASE_VEHICLE_IDS and side == "vehicle":
                count += 1
                all_boxes.append((x1,y1,x2,y2)); box_types.append('vehicle')
                if draw:
                    lbl = {2:"car",3:"moto",5:"bus",7:"truck"}.get(cls_id,"veh")
                    cv2.rectangle(out,(x1,y1),(x2,y2),(0,210,255),2)
                    _label(out, f"{lbl}(b) {conf:.2f}", x1, y1, (0,210,255))
                drawn_boxes.append((x1,y1,x2,y2))

    # ── [M1] Motion tagging ──
    if motion_tracker and all_boxes:
        moving_idx = motion_tracker.moving_boxes(frame, all_boxes)
        for i, (x1,y1,x2,y2) in enumerate(all_boxes):
            if i not in moving_idx:
                stationary += 1
                if draw:
                    cv2.putText(out,"STATIC",(x1,y1-2),
                                cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,180,255),1,cv2.LINE_AA)

    # ── [M4] Pedestrian speed ──
    if side == "pedestrian":
        centers = [((x1+x2)//2, (y1+y2)//2) for (x1,y1,x2,y2) in all_boxes]
        ped_speed_tracker.update(centers)
        slow_ped = ped_speed_tracker.is_slow() and count > 0

    # ── Night mode indicator ──
    if draw and is_low_light(frame):
        cv2.putText(out,"[NIGHT MODE]",(8, FEED_H-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,180,255),1,cv2.LINE_AA)

    return out, count, disabled, emergency, stationary, slow_ped


def _overlaps(x1,y1,x2,y2, boxes, iou_thresh=0.40):
    for bx1,by1,bx2,by2 in boxes:
        ix1,iy1 = max(x1,bx1), max(y1,by1)
        ix2,iy2 = min(x2,bx2), min(y2,by2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter == 0:
            continue
        union = (x2-x1)*(y2-y1)+(bx2-bx1)*(by2-by1)-inter
        if inter/union > iou_thresh:
            return True
    return False

# ── Snapshot ───────────────────────────────────────
def snapshot(cap, side, motion_tracker=None):
    person_votes = vehicle_votes = disabled_vote = emrg_vote = 0
    last_frame = None

    for _ in range(SNAP_FRAMES):
        ret, f = cap.read()
        if not ret:
            # [L2] Graceful loop — don't reset timer, just rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, f = cap.read()
        if not ret:
            break
        f = cv2.resize(f, (FEED_W, FEED_H))
        last_frame = f.copy()
        _, cnt, dis, emrg, _, _ = detect(f, side, draw=False,
                                          motion_tracker=motion_tracker)
        if side == "pedestrian" and cnt > 0: person_votes  += 1
        if side == "vehicle"    and cnt > 0: vehicle_votes += 1
        if dis:   disabled_vote += 1
        if emrg:  emrg_vote     += 1

    if last_frame is None:
        return np.zeros((FEED_H,FEED_W,3),dtype=np.uint8), 0, False, False, 0, False

    annotated, count, disabled, emrg, stat, slow = detect(
        last_frame, side, draw=True, motion_tracker=motion_tracker)

    if side == "pedestrian" and person_votes  < SNAP_NEEDED: count = 0
    if side == "vehicle"    and vehicle_votes < SNAP_NEEDED: count = 0
    if disabled_vote < 2: disabled = False
    if emrg_vote     < 2: emrg     = False

    print(f"  [SNAP:{side}] count={count} disabled={disabled} emergency={emrg} "
          f"stationary={stat} slow_ped={slow}")
    return annotated, count, disabled, emrg, stat, slow

# ── Timer logic ────────────────────────────────────
def calc_vehicle_timer(count):
    if count > 8:  return 15
    if count >= 4: return 10
    return 5

def calc_ped_timer(count, disabled, slow_ped=False):
    base = 5
    if count >= 5:  base = 8
    if count > 10:  base = 15
    if disabled:    base += 5    # [M4] disabled extra
    if slow_ped:    base += 3    # [M4] slow walkers extra
    return base

# ── Lane State ─────────────────────────────────────
class LaneState:
    def __init__(self, name, side):
        self.name            = name
        self.side            = side
        self.green           = False
        self.timer           = 0
        self.max_timer       = 1
        self.count           = 0
        self.disabled        = False
        self.emergency_lock  = False
        self.last_tick       = time.time()
        self.bar_progress    = 0.0
        self.stationary      = 0
        self.slow_ped        = False
        # [M3] wait time per lane
        self.total_wait_secs = 0.0
        self.red_since       = None
        # [L4] starvation: track how long this lane has been red
        self.red_start       = None

    def start_green(self, count, disabled, stat=0, slow_ped=False):
        self.count        = count
        self.disabled     = disabled
        self.stationary   = stat
        self.slow_ped     = slow_ped
        self.timer        = (calc_vehicle_timer(count)
                             if self.side == "vehicle"
                             else calc_ped_timer(count, disabled, slow_ped))
        self.max_timer    = self.timer
        self.bar_progress = 1.0
        self.last_tick    = time.time()
        self.green        = True
        self.red_start    = None
        print(f"  [GREEN] {self.name}({self.side}): count={count} "
              f"disabled={disabled} slow_ped={slow_ped} timer={self.timer}s")

    def stop(self):
        self.green          = False
        self.emergency_lock = False
        self.bar_progress   = 0.0
        if self.red_start is None:
            self.red_start  = time.time()

    def tick(self):
        if not self.green:
            return False
        if self.emergency_lock:
            self.last_tick = time.time()
            return False
        now = time.time()
        if now - self.last_tick >= 1.0:
            self.timer    -= 1
            self.last_tick = now
            self.bar_progress = max(0.0, self.timer / max(self.max_timer, 1))
            if self.timer <= 0:
                return True
        return False

    def red_wait(self):
        """Seconds this lane has been waiting at red."""
        if self.red_start and not self.green:
            return time.time() - self.red_start
        return 0.0

    def starvation_check(self):
        """[L4] Returns True if this lane has waited too long."""
        return self.red_wait() > MAX_WAIT_CAP

# ── Auto-classify lane ─────────────────────────────
def classify_lane(cap, n_frames=10):
    person_total = vehicle_total = 0
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for _ in range(n_frames):
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (640, 480))
        res = base_model(f, conf=0.35, verbose=False, augment=False)
        for r in res:
            for box in r.boxes:
                cid = int(box.cls[0])
                if cid == BASE_PERSON_ID:   person_total  += 1
                elif cid in BASE_VEHICLE_IDS: vehicle_total += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    side = "pedestrian" if person_total >= vehicle_total else "vehicle"
    print(f"  [AUTO-CLASSIFY] persons={person_total} vehicles={vehicle_total} => '{side}'")
    return side

# ── Traffic light drawing ───────────────────────────
def draw_light(panel, green, emrg_lock, label_text):
    h, w = panel.shape[:2]
    cx   = w // 2
    top  = h // 2 - 120

    cv2.rectangle(panel,(cx-32,top-2),(cx+32,top+187),(55,55,68),-1,cv2.LINE_AA)
    cv2.rectangle(panel,(cx-30,top),(cx+30,top+185),(32,32,44),-1,cv2.LINE_AA)

    bulbs = [
        (cx, top+32,  RED_ON    if not green    else DIM_BULB),
        (cx, top+92,  ORANGE_ON if emrg_lock    else DIM_BULB),
        (cx, top+152, GREEN_ON  if green         else DIM_BULB),
    ]
    for bx, by, col in bulbs:
        if col != DIM_BULB:
            for r, a in [(24,0.10),(20,0.20)]:
                ov = panel.copy()
                cv2.circle(ov,(bx,by),r,col,-1,cv2.LINE_AA)
                cv2.addWeighted(ov,a,panel,1-a,0,panel)
        cv2.circle(panel,(bx,by),13,col,-1,cv2.LINE_AA)
        cv2.circle(panel,(bx,by),13,(0,0,0),1,cv2.LINE_AA)

    if emrg_lock: state_txt, state_col = "EMRG", ORANGE_ON
    elif green:   state_txt, state_col = "GO",   GREEN_ON
    else:         state_txt, state_col = "STOP", RED_ON

    tw = cv2.getTextSize(state_txt,cv2.FONT_HERSHEY_DUPLEX,0.65,1)[0][0]
    cv2.putText(panel, state_txt,(cx-tw//2,top+210),
                cv2.FONT_HERSHEY_DUPLEX,0.65,state_col,1,cv2.LINE_AA)

    lw = cv2.getTextSize(label_text,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)[0][0]
    cv2.putText(panel, label_text,(cx-lw//2,h-12),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,GRAY,1,cv2.LINE_AA)


def badge(img, text, x, y, bg, fg=(12,12,20)):
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.50,1)
    cv2.rectangle(img,(x,y),(x+tw+16,y+th+8),bg,-1,cv2.LINE_AA)
    cv2.putText(img,text,(x+8,y+th+2),
                cv2.FONT_HERSHEY_SIMPLEX,0.50,fg,1,cv2.LINE_AA)

# ── [M2] Queue length bar ───────────────────────────
def draw_queue_bar(screen, x1, x2, y, count, max_count=20, label=""):
    """Visual bar showing queue density."""
    bar_w = x2 - x1 - 20
    fill  = int(bar_w * min(count / max_count, 1.0))
    ratio = count / max_count
    col   = GREEN_ON if ratio < 0.4 else (ORANGE_ON if ratio < 0.75 else RED_ON)
    cv2.rectangle(screen,(x1+10,y),(x1+10+bar_w,y+10),(38,38,50),-1)
    if fill > 0:
        cv2.rectangle(screen,(x1+10,y),(x1+10+fill,y+10),col,-1)
    cv2.putText(screen,f"Queue:{count}",(x1+10,y+24),
                cv2.FONT_HERSHEY_SIMPLEX,0.38,GRAY,1,cv2.LINE_AA)

# ── Build screen ────────────────────────────────────
def build_screen(f1, f2, lane1, lane2):
    screen = np.full((TOTAL_H,TOTAL_W,3), BG, dtype=np.uint8)

    # ── Cool header ──────────────────────────────────
    screen[:HEADER,:] = DARK
    # Subtle horizontal gradient overlay on header
    for x in range(TOTAL_W):
        t = x / TOTAL_W
        b_val = int(18 + 20 * (1 - abs(t - 0.5) * 2))
        screen[:HEADER, x] = (b_val, b_val + 4, b_val + 10)

    # Accent line at bottom of header
    screen[HEADER-3:HEADER, :] = (0, 180, 220)

    # Side accent dots
    for dx in [0, 1, 2]:
        cv2.circle(screen, (18 + dx*18, HEADER//2), 5,
                   [(0,210,255),(0,160,200),(0,100,150)][dx], -1, cv2.LINE_AA)
        cv2.circle(screen, (TOTAL_W - 18 - dx*18, HEADER//2), 5,
                   [(0,210,255),(0,160,200),(0,100,150)][dx], -1, cv2.LINE_AA)

    # Main title
    title    = "SMART TRAFFIC MANAGEMENT SYSTEM"
    subtitle = "by KHUSHI & SWATI"
    tw = cv2.getTextSize(title,    cv2.FONT_HERSHEY_DUPLEX, 0.80, 2)[0][0]
    sw = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX,0.38, 1)[0][0]
    # Shadow
    cv2.putText(screen, title, (TOTAL_W//2 - tw//2 + 2, 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.80, (0,60,80), 2, cv2.LINE_AA)
    # Main text (cyan-white)
    cv2.putText(screen, title, (TOTAL_W//2 - tw//2, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.80, (200, 240, 255), 2, cv2.LINE_AA)
    # Subtitle
    cv2.putText(screen, subtitle, (TOTAL_W//2 - sw//2, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 180, 200), 1, cv2.LINE_AA)

    lp = screen[HEADER:HEADER+FEED_H, 0:PANEL_W]
    lp[:] = PANEL_BG
    draw_light(lp, lane1.green, lane1.emergency_lock, lane1.side.upper())

    rp = screen[HEADER:HEADER+FEED_H, PANEL_W+FEED_W+GAP+FEED_W:]
    rp[:] = PANEL_BG
    draw_light(rp, lane2.green, lane2.emergency_lock, lane2.side.upper())

    def fc(ln):
        if ln.emergency_lock: return ORANGE_ON
        return GREEN_ON if ln.green else RED_ON

    b = 2
    screen[HEADER-b:HEADER+FEED_H+b, PANEL_W-b:PANEL_W+FEED_W+b] = fc(lane1)
    screen[HEADER-b:HEADER+FEED_H+b,
           PANEL_W+FEED_W+GAP-b:PANEL_W+FEED_W+GAP+FEED_W+b] = fc(lane2)

    screen[HEADER:HEADER+FEED_H, PANEL_W:PANEL_W+FEED_W] = f1
    screen[HEADER:HEADER+FEED_H,
           PANEL_W+FEED_W+GAP:PANEL_W+FEED_W+GAP+FEED_W] = f2

    col1 = (0,210,255) if lane1.side=="vehicle" else (60,220,80)
    col2 = (0,210,255) if lane2.side=="vehicle" else (60,220,80)
    cv2.putText(screen, f"{lane1.side.upper()} LANE",
                (PANEL_W+10, HEADER+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col1, 1, cv2.LINE_AA)
    cv2.putText(screen, f"{lane2.side.upper()} LANE",
                (PANEL_W+FEED_W+GAP+10, HEADER+22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col2, 1, cv2.LINE_AA)

    fy = HEADER + FEED_H
    screen[fy:fy+FOOTER,:] = DARK
    screen[fy:fy+2,:] = DIVIDER

    VX1,VX2 = PANEL_W, PANEL_W+FEED_W
    PX1,PX2 = PANEL_W+FEED_W+GAP, PANEL_W+FEED_W+GAP+FEED_W
    BY1,BY2 = fy+14, fy+30

    # Timer bars
    for (x1,x2), ln in [((VX1,VX2),lane1), ((PX1,PX2),lane2)]:
        cv2.rectangle(screen,(x1,BY1),(x2,BY2),(38,38,50),-1)
        fill = int((x2-x1)*ln.bar_progress)
        col  = ORANGE_ON if ln.emergency_lock else (GREEN_ON if ln.green else RED_ON)
        if fill > 0:
            cv2.rectangle(screen,(x1,BY1),(x1+fill,BY2),col,-1)
        txt = "EMRG HOLD" if ln.emergency_lock else f"{max(ln.timer,0)}s"
        tw  = cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,0.50,1)[0][0]
        cv2.putText(screen,txt,((x1+x2)//2-tw//2,BY2-1),
                    cv2.FONT_HERSHEY_DUPLEX,0.50,WHITE,1,cv2.LINE_AA)

    # [M2] Queue bars
    draw_queue_bar(screen, VX1, VX2, fy+36, lane1.count)
    draw_queue_bar(screen, PX1, PX2, fy+36, lane2.count)

    # Count + stationary badges
    def lane_badge_txt(ln):
        if ln.side == "vehicle":
            t = f"Vehicles:{ln.count}"
            if ln.stationary: t += f" ({ln.stationary} static)"
        else:
            t = f"People:{ln.count}"
            if ln.disabled:   t += " [+DISABLED +5s]"
            if ln.slow_ped:   t += " [SLOW +3s]"
        return t

    l1_bg = ACCENT if lane1.side=="vehicle" else (60,180,60)
    l2_bg = ACCENT if lane2.side=="vehicle" else (60,180,60)
    l1_fg = (10,10,10) if lane1.side=="vehicle" else WHITE
    l2_fg = (10,10,10) if lane2.side=="vehicle" else WHITE

    badge(screen, lane_badge_txt(lane1), VX1+10, fy+58, l1_bg, l1_fg)
    badge(screen, lane_badge_txt(lane2), PX1+10, fy+58, l2_bg, l2_fg)

    # [M3] Total wait time display
    cv2.putText(screen,
                f"Wait: {int(lane1.red_wait())}s / cap {MAX_WAIT_CAP}s",
                (VX1+10, fy+86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, PURPLE, 1, cv2.LINE_AA)
    cv2.putText(screen,
                f"Wait: {int(lane2.red_wait())}s / cap {MAX_WAIT_CAP}s",
                (PX1+10, fy+86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, PURPLE, 1, cv2.LINE_AA)

    # Emergency / starvation banners
    for (lnx, ln) in [(VX1, lane1),(PX1, lane2)]:
        if ln.emergency_lock:
            badge(screen," !! EMERGENCY - GREEN HELD !! ",
                  lnx+10, fy+92, ORANGE_ON,(10,10,10))
        elif ln.starvation_check():
            badge(screen," STARVATION - FORCE SWITCH! ",
                  lnx+10, fy+92, (0,0,200), WHITE)
        elif ln.disabled:
            badge(screen," DISABLED DETECTED - +5s ",
                  lnx+10, fy+92,(40,40,220),WHITE)

    # Status icon — compact, no redundant RED/GREEN text
    for (lnx2, ln) in [(VX1, lane1),(PX1, lane2)]:
        if ln.green and not ln.emergency_lock:
            dot_col = GREEN_ON
            dot_txt = f"  ACTIVE  {max(ln.timer,0)}s left"
        elif ln.emergency_lock:
            dot_col = ORANGE_ON
            dot_txt = "  EMERGENCY HOLD"
        else:
            dot_col = (80, 80, 100)
            dot_txt = f"  Waiting {int(ln.red_wait())}s"
        cv2.circle(screen, (lnx2+16, fy+112), 6, dot_col, -1, cv2.LINE_AA)
        cv2.putText(screen, dot_txt, (lnx2+10, fy+117),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, dot_col, 1, cv2.LINE_AA)

    return screen

# ==================================================
#  MAIN
# ==================================================
print("=" * 55)
print("  AI Smart Traffic Management  |  v5")
print("=" * 55)

print("\n[AUTO-CLASSIFY] Analysing lane1 (lane3.mp4)...")
side1 = classify_lane(vehicle_cap)
print(f"[AUTO-CLASSIFY] Analysing lane2 (lane2.mp4)...")
side2 = classify_lane(ped_cap)
print(f"\n  lane3.mp4 => {side1}")
print(f"  lane2.mp4 => {side2}\n")

lane1 = LaneState("Lane-1", side1)
lane2 = LaneState("Lane-2", side2)

print("[BOOT] Snapshotting Lane-1...")
snap1, c1, dis1, emrg1, stat1, slow1 = snapshot(vehicle_cap, side1, motion1)
lane1.start_green(c1, dis1, stat1, slow1)
lane1.emergency_lock = emrg1
lane2.stop()

disp1 = snap1.copy()
disp2 = np.zeros((FEED_H,FEED_W,3), dtype=np.uint8)

print("\n[LOOP] Starting — press ESC to quit\n")

while True:
    ret1, frame1 = vehicle_cap.read()
    ret2, frame2 = ped_cap.read()

    # [L2] Graceful loop — no timer reset
    if not ret1:
        vehicle_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, frame1 = vehicle_cap.read()
    if not ret2:
        ped_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret2, frame2 = ped_cap.read()

    frame1 = cv2.resize(frame1,(FEED_W,FEED_H))
    frame2 = cv2.resize(frame2,(FEED_W,FEED_H))

    # Live detection
    disp1, l1_cnt, l1_dis, l1_emrg, l1_stat, l1_slow = detect(
        frame1, side1, motion_tracker=motion1)
    disp2, l2_cnt, l2_dis, l2_emrg, l2_stat, l2_slow = detect(
        frame2, side2, motion_tracker=motion2)

    # Emergency lock on active lane
    if lane1.green: lane1.emergency_lock = l1_emrg
    if lane2.green: lane2.emergency_lock = l2_emrg

    # [L1] Emergency priority queue: vehicle EV > pedestrian EV
    # If BOTH lanes have emergency, vehicle lane wins
    both_emrg = l1_emrg and l2_emrg
    if both_emrg:
        # Force vehicle lane green regardless
        vehicle_lane  = lane1 if side1 == "vehicle" else lane2
        ped_lane      = lane1 if side1 == "pedestrian" else lane2
        v_cap         = vehicle_cap if side1 == "vehicle" else ped_cap
        p_cap         = vehicle_cap if side1 == "pedestrian" else ped_cap
        v_mt          = motion1     if side1 == "vehicle" else motion2

        if not vehicle_lane.green:
            print("[L1] Both-emergency: vehicle lane wins priority")
            ped_lane.stop()
            sv, cv_, dv, _, stv, slv = snapshot(v_cap, "vehicle", v_mt)
            if side1 == "vehicle": disp1 = sv
            else:                  disp2 = sv
            vehicle_lane.start_green(cv_, dv, stv, slv)
            vehicle_lane.emergency_lock = True

    else:
        # Single-lane emergency override
        if l1_emrg and not lane1.green and not lane1.emergency_lock:
            print("[!] EMERGENCY Lane-1 (RED) - overriding to GREEN")
            lane2.stop()
            s1, c1, _, _, st1, sl1 = snapshot(vehicle_cap, side1, motion1)
            disp1 = s1
            lane1.start_green(c1, False, st1, sl1)
            lane1.emergency_lock = True

        elif l2_emrg and not lane2.green and not lane2.emergency_lock:
            print("[!] EMERGENCY Lane-2 (RED) - overriding to GREEN")
            lane1.stop()
            s2, c2, d2, _, st2, sl2 = snapshot(ped_cap, side2, motion2)
            disp2 = s2
            lane2.start_green(c2, d2, st2, sl2)
            lane2.emergency_lock = True

    # [L4] Starvation check — force switch if waiting too long
    if lane1.starvation_check() and not lane1.green:
        print(f"[L4] STARVATION Lane-1 — waited >{MAX_WAIT_CAP}s, forcing green")
        lane2.stop()
        s1, c1, d1, e1, st1, sl1 = snapshot(vehicle_cap, side1, motion1)
        disp1 = s1
        lane1.start_green(c1, d1, st1, sl1)
        if e1: lane1.emergency_lock = True

    elif lane2.starvation_check() and not lane2.green:
        print(f"[L4] STARVATION Lane-2 — waited >{MAX_WAIT_CAP}s, forcing green")
        lane1.stop()
        s2, c2, d2, e2, st2, sl2 = snapshot(ped_cap, side2, motion2)
        disp2 = s2
        lane2.start_green(c2, d2, st2, sl2)
        if e2: lane2.emergency_lock = True

    # Tick
    exp1 = lane1.tick()
    exp2 = lane2.tick()

    # Normal switch on expiry
    if exp1:
        print("[SWITCH] Lane-1 expired -> Lane-2 green")
        lane1.stop()
        s2, c2, d2, e2, st2, sl2 = snapshot(ped_cap, side2, motion2)
        disp2 = s2
        lane2.start_green(c2, d2, st2, sl2)
        if e2: lane2.emergency_lock = True

    if exp2:
        print("[SWITCH] Lane-2 expired -> Lane-1 green")
        lane2.stop()
        s1, c1, d1, e1, st1, sl1 = snapshot(vehicle_cap, side1, motion1)
        disp1 = s1
        lane1.start_green(c1, d1, st1, sl1)
        if e1: lane1.emergency_lock = True

    screen = build_screen(disp1, disp2, lane1, lane2)
    cv2.imshow("Smart Tracffic Control System", screen)

    if cv2.waitKey(1) == 27:
        break

vehicle_cap.release()
ped_cap.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
