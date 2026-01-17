import os
import json
import time
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

BRICK_MODEL_PATH = "lego_mobilenetv2_288_ft.tflite"
BRICK_LABELS_PATH = "class_names.json"

COLOR_MODEL_PATH = "lego_color_128.tflite"     
COLOR_LABELS_PATH = "color_names.json"     

YOLO_MODEL_PATH = "best.pt"

TTA_REPEATS = 3
LOW_CONF_THRESHOLD = 0.55

detected_objects = []
selected_id = None
selected_point = None 

def load_interpreter(model_path: str):
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    intrp = Interpreter(model_path=model_path)
    intrp.allocate_tensors()
    return intrp

def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list):
        raise ValueError("Plik etykiet musi byc lista stringow (JSON list)")
    return names

def preprocess_bgr(img_bgr, size=(288, 288)):
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(x, axis=0)

def color_preprocess_bgr(img_bgr, size):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.expand_dims(x, axis=0)

def softmax(logits):
    x = logits - np.max(logits, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)

def apply_low_conf_rules(p, class_names, threshold=LOW_CONF_THRESHOLD):
    top2 = np.argsort(p)[-2:][::-1]
    c1, c2 = class_names[top2[0]], class_names[top2[1]]
    if p[top2[0]] < threshold and ("3005 brick 1x1" in (c1, c2)):
        if "3005 brick 1x1" in class_names:
            return class_names.index("3005 brick 1x1"), p
    return top2[0], p

def tta_predict(interp, input_index, output_index, img_bgr, size, repeats=TTA_REPEATS):
    if repeats <= 1:
        x = preprocess_bgr(img_bgr, (size, size))
        interp.set_tensor(input_index, x)
        interp.invoke()
        return interp.get_tensor(output_index)[0]

    augs = [
        img_bgr,
        cv2.flip(img_bgr, 1),
        cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.rotate(img_bgr, cv2.ROTATE_180),
    ][:repeats]

    probs = []
    for im in augs:
        x = preprocess_bgr(im, (size, size))
        interp.set_tensor(input_index, x)
        interp.invoke()
        logits = interp.get_tensor(output_index)[0]
        probs.append(softmax(logits))
    probs_mean = np.mean(probs, axis=0)
    return np.log(np.clip(probs_mean, 1e-9, 1.0))

from ultralytics import YOLO  # type: ignore
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"Nie znaleziono modelu YOLO:\n{YOLO_MODEL_PATH}")
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

def detect_brick_boxes_yolo(img_bgr, conf_thr=0.25):
    model = get_yolo_model()
    res = model(
        img_bgr,
        verbose=False,
        conf=conf_thr,
        iou=0.5,
        imgsz=1280,
        max_det=50,
        agnostic_nms=True
    )[0]

    objs = []
    if len(res.boxes) == 0:
        return objs

    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        conf = float(b.conf[0].cpu().item())
        objs.append((int(x1), int(y1), int(x2), int(y2), conf))

    objs = merge_boxes_xyxy(
        objs,
        iou_thr=0.25,       
        contain_thr=0.70,    
        max_passes=2
    )
    return objs

def crop_roi(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)
    area = max(1, bw * bh)

    pad = 0.02 if area < 8000 else 0.10
    px = int(bw * pad)
    py = int(bh * pad)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return roi

    if area < 8000:
        roi = cv2.resize(roi, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)

    return roi

def pick_by_bbox_center(objs, pt, inner_ratio=0.6):
    if not objs or pt is None:
        return None

    x, y = pt
    best = None
    best_d = 10**18

    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        half_w = (bw * inner_ratio) / 2.0
        half_h = (bh * inner_ratio) / 2.0

        ix1 = cx - half_w
        ix2 = cx + half_w
        iy1 = cy - half_h
        iy2 = cy + half_h

        if not (ix1 <= x <= ix2 and iy1 <= y <= iy2):
            continue

        d = (cx - x) ** 2 + (cy - y) ** 2
        if d < best_d:
            best_d = d
            best = o

    return best

class InferenceWorker(threading.Thread):
    def __init__(self, app, frame_queue):
        super().__init__(daemon=True)
        self.app = app
        self.frame_queue = frame_queue
        self.running = False
        self.paused = False
        self.cap = None

    def open_camera(self, idx: int) -> bool:
        self.release()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ok, _ = self.cap.read()
        return ok

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    def stop(self):
        self.running = False
        self.release()

    def run(self):
        global detected_objects, selected_id

        self.running = True
        t_prev = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            self.app._last_frame_bgr = frame.copy()

            try:
                boxes = detect_brick_boxes_yolo(frame, conf_thr=0.25)
            except Exception as e:
                print("YOLO error:", repr(e))
                boxes = []

            objs = []
            
            for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
                roi = crop_roi(frame, (x1, y1, x2, y2))
                if roi is None or roi.size == 0:
                    continue

                logits = tta_predict(
                    self.app.interp,
                    self.app.input_index,
                    self.app.output_index,
                    roi,
                    size=self.app.model_input_size,
                    repeats=TTA_REPEATS
                )
                p_type = softmax(logits)
                _, p_type = apply_low_conf_rules(p_type, self.app.class_names)
                top_ids = np.argsort(p_type)[-5:][::-1]

                color_name = None
                p_color = None
                if self.app.color_interp is not None and self.app.color_class_names:
                    x = color_preprocess_bgr(roi, self.app.color_input_size)
                    self.app.color_interp.set_tensor(self.app.color_input_index, x)
                    self.app.color_interp.invoke()
                    logits_c = self.app.color_interp.get_tensor(self.app.color_output_index)[0]
                    p_color = softmax(logits_c)
                    color_name = self.app.color_class_names[int(np.argmax(p_color))]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                objs.append({
                    "id": i,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "p_det": conf,
                    "type_top_ids": top_ids,
                    "p_type": p_type,
                    "type_name": self.app.class_names[int(top_ids[0])],
                    "type_p": float(p_type[int(top_ids[0])]),
                    "color_name": color_name,
                    "p_color": p_color
                })

            detected_objects = objs
            active = None
            focus_only = bool(self.app.var_focus_selected.get())

            if detected_objects:
                if selected_id is not None:
                    for obj in detected_objects:
                        if obj["id"] == selected_id:
                            active = obj
                            break

                if (not focus_only) and (active is None):
                    active = max(detected_objects, key=lambda o: o["p_det"])
                    selected_id = active["id"]

            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t_prev))
            t_prev = t1
            vis = self.app.draw_overlay(frame, fps=fps, objs=detected_objects, active=active)
            payload = (vis, detected_objects, active)
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(payload)
                except Exception:
                    pass

def overlay_scale_params(h: int, w: int): # type: ignore
    base = 720.0
    s = max(0.6, min(2.2, min(w, h) / base))  

    panel_h = int(60 * s)
    thickness = max(1, int(2 * s))
    text_scale = 0.5 * s
    text_thickness = max(1, int(2 * s))
    click_r1 = max(4, int(6 * s))
    click_r2 = max(8, int(10 * s))

    return {
        "s": s,
        "panel_h": panel_h,
        "thickness": thickness,
        "text_scale": text_scale,
        "text_thickness": text_thickness,
        "click_r1": click_r1,
        "click_r2": click_r2,
    }

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter + 1e-9)

def bbox_containment(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a), inter / float(area_b)

def union_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))

def merge_boxes_xyxy(
    boxes,
    iou_thr=0.25,
    contain_thr=0.70,
    max_passes=2
):
    if not boxes:
        return []

    merged = boxes[:]
    for _ in range(max_passes):
        merged = sorted(merged, key=lambda x: x[4], reverse=True)
        out = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            x1, y1, x2, y2, conf = merged[i]
            cur_bbox = (x1, y1, x2, y2)
            cur_conf = conf

            used[i] = True

            changed = True
            while changed:
                changed = False
                for j in range(len(merged)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2, bconf = merged[j]
                    b_bbox = (bx1, by1, bx2, by2)

                    iou = bbox_iou(cur_bbox, b_bbox)
                    a_in_b, b_in_a = bbox_containment(cur_bbox, b_bbox)

                    if (iou >= iou_thr) or (a_in_b >= contain_thr) or (b_in_a >= contain_thr):
                        cur_bbox = union_bbox(cur_bbox, b_bbox)
                        cur_conf = max(cur_conf, bconf)
                        used[j] = True
                        changed = True

            out.append((int(cur_bbox[0]), int(cur_bbox[1]), int(cur_bbox[2]), int(cur_bbox[3]), float(cur_conf)))

        if len(out) == len(merged):
            merged = out
            break

        merged = out

    return merged

def nms_boxes_xyxy(boxes, iou_thr=0.35):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)

        new_boxes = []
        for b in boxes:
            iou = bbox_iou(best[:4], b[:4])
            if iou < iou_thr:
                new_boxes.append(b)
        boxes = new_boxes

    return keep

class LegoStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LEGO Studio")
        self.geometry("1120x720")
        self.minsize(960, 640)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#1f1f28")
        style.configure("TLabel", background="#1f1f28", foreground="#f5f5f5")
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#ffd447")
        style.configure("Status.TLabel", background="#15151c", foreground="#d0d0d0")
        style.configure("TButton", padding=4)
        self.configure(bg="#1f1f28")
        self.interp = None
        self.input_index = None
        self.output_index = None
        self.class_names = []
        self.model_input_size = 288
        self.color_interp = None
        self.color_input_index = None
        self.color_output_index = None
        self.color_class_names = []
        self.color_input_size = 96
        self.worker = None
        self.frame_queue = queue.Queue(maxsize=2)
        self._preview_imgtk = None
        self._last_vis_frame = None
        self._last_frame_bgr = None  
        self.var_camera_idx = tk.IntVar(value=0)
        self.var_paused = tk.BooleanVar(value=False)
        self.var_focus_selected = tk.BooleanVar(value=False)
        self.var_focus_selected.trace_add("write", lambda *_: self._clear_selection())
        self._build_layout()
        self.preview.bind("<Button-1>", self._on_preview_click)
        self._load_models_startup()
        self.after(15, self._ui_loop)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_models_startup(self):
        if os.path.exists(BRICK_MODEL_PATH) and os.path.exists(BRICK_LABELS_PATH):
            try:
                self._load_brick_model()
                self.status.config(text=f"Zaladowano model klockow ({self.model_input_size}x{self.model_input_size}), klasy: {len(self.class_names)}")
            except Exception as e:
                self.status.config(text=f"Blad ladowania modelu klockow: {e}")
        else:
            self.status.config(text="Brak modelu klockow lub etykiet - sprawdz pliki.")

        try:
            self._load_color_model()
        except Exception as e:
            print("Model koloru nie zostal zaladowany:", e)

    def _load_brick_model(self):
        self.interp = load_interpreter(BRICK_MODEL_PATH)
        ids_in = self.interp.get_input_details()
        ids_out = self.interp.get_output_details()
        self.input_index = ids_in[0]["index"]
        self.output_index = ids_out[0]["index"]
        h, w = int(ids_in[0]["shape"][1]), int(ids_in[0]["shape"][2])
        if h != w:
            raise ValueError(f"Model ma niesquare input: {h}x{w}")
        self.model_input_size = h
        self.class_names = load_labels(BRICK_LABELS_PATH)

    def _load_color_model(self):
        if not (os.path.exists(COLOR_MODEL_PATH) and os.path.exists(COLOR_LABELS_PATH)):
            print("Model koloru lub etykiety nie znalezione.")
            return
        self.color_interp = load_interpreter(COLOR_MODEL_PATH)
        ids_in = self.color_interp.get_input_details()
        ids_out = self.color_interp.get_output_details()
        self.color_input_index = ids_in[0]["index"]
        self.color_output_index = ids_out[0]["index"]
        h, w = int(ids_in[0]["shape"][1]), int(ids_in[0]["shape"][2])
        if h != w:
            raise ValueError(f"Model koloru ma niesquare input: {h}x{w}")
        self.color_input_size = h
        self.color_class_names = load_labels(COLOR_LABELS_PATH)

    def _build_layout(self):
        root_frame = ttk.Frame(self)
        root_frame.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(root_frame)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        ttk.Label(top, text="LEGO Studio", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(top, text="Kamera:", padding=(15, 0, 4, 0)).pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=0, to=9, width=4, textvariable=self.var_camera_idx).pack(side=tk.LEFT)
        ttk.Button(top, text="Start kamera", command=self._start_camera).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Pauza / Wznów", command=self._toggle_pause).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(top, text="Wybór klocka", variable=self.var_focus_selected).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="Wyczyść wybór", command=self._clear_selection).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Otwórz obraz…", command=self._open_image_and_run).pack(side=tk.RIGHT, padx=4)
        mid = ttk.Frame(root_frame)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.preview = tk.Label(mid, bg="#050509")
        self.preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right = ttk.Frame(mid, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(right, text="Wynik", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.result_text = tk.Text(
            right, height=10, width=40,
            bg="#15151c", fg="#f5f5f5",
            relief=tk.FLAT
        )
        self.result_text.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(right, text="Top-5 (dla wybranego)", padding=(0, 4, 0, 2)).pack(anchor="w")
        self.bars_frame = ttk.Frame(right)
        self.bars_frame.pack(fill=tk.X, pady=(0, 8))
        self._bars = []
        for _ in range(5):
            row = ttk.Frame(self.bars_frame)
            row.pack(fill=tk.X, pady=2)
            lbl = ttk.Label(row, text="—", width=22)
            lbl.pack(side=tk.LEFT)
            bar = ttk.Progressbar(row, orient=tk.HORIZONTAL, maximum=1.0, length=160, mode="determinate")
            bar.pack(side=tk.RIGHT, padx=6)
            self._bars.append((lbl, bar))

        self.status = ttk.Label(self, text="Gotowe", style="Status.TLabel", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _start_camera(self):
        if self.interp is None or not self.class_names:
            messagebox.showwarning("Uwaga", "Brak zaladowanego modelu klockow lub etykiet.")
            return

        self._stop()
        self.worker = InferenceWorker(self, self.frame_queue)
        ok = self.worker.open_camera(self.var_camera_idx.get())
        if not ok:
            self.worker = None
            messagebox.showerror("Blad", "Nie moge otworzyc kamery. Sprobuj inny index.")
            return

        self.worker.start()
        self.var_paused.set(False)
        self.status.config(text="Kamera uruchomiona.")

    def _toggle_pause(self):
        if self.worker is None:
            return
        self.worker.paused = not self.worker.paused
        self.var_paused.set(self.worker.paused)
        self.status.config(text="Wstrzymano podglad." if self.worker.paused else "Wznowiono podglad.")

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
            self.status.config(text="Zatrzymano kamerę.")

    def _clear_selection(self):
        global selected_id, selected_point
        selected_id = None
        selected_point = None
        self.status.config(text="Wyczyszczono wybor.")
        self._refresh_view_after_selection()

    def _refresh_view_after_selection(self):
        global detected_objects, selected_id

        if self._last_frame_bgr is None:
            return

        active = None
        if selected_id is not None:
            for o in detected_objects:
                if o["id"] == selected_id:
                    active = o
                    break

        vis = self.draw_overlay(self._last_frame_bgr, fps=0.0, objs=detected_objects, active=active)
        payload = (vis, detected_objects, active)
        try:
            self.frame_queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self.frame_queue.put_nowait(payload)
            except Exception:
                pass

    def _on_preview_click(self, event):
        global selected_point, selected_id, detected_objects

        if self._last_vis_frame is None:
            return

        vis = self._last_vis_frame
        ih, iw = vis.shape[:2]

        lw = self.preview.winfo_width() or 640
        lh = self.preview.winfo_height() or 480
        scale = min(lw / iw, lh / ih)

        nw, nh = int(iw * scale), int(ih * scale)
        ox = (lw - nw) // 2
        oy = (lh - nh) // 2

        if not (ox <= event.x < ox + nw and oy <= event.y < oy + nh):
            return

        x = int((event.x - ox) / scale)
        y = int((event.y - oy) / scale)

        panel_h = overlay_scale_params(ih, iw)["panel_h"]
        if y >= ih - panel_h:
            return

        selected_point = (x, y)

        if self.var_focus_selected.get():
            hit = pick_by_bbox_center(detected_objects, selected_point, inner_ratio=0.6)
            if hit is not None:
                selected_id = hit["id"]
                self.status.config(text=f"Wybrano ID {selected_id}")
            else:
                selected_id = None
                self.status.config(text=f"Klik: {selected_point} (nie trafiono w srodek klocka)")

            self._refresh_view_after_selection()

    def _open_image_and_run(self):
        if self.interp is None or not self.class_names:
            messagebox.showwarning("Uwaga", "Brak zaladowanego modelu klockow lub etykiet.")
            return

        self._stop()

        p = filedialog.askopenfilename(
            title="Wybierz obraz z klockami",
            filetypes=[("Obrazy", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
        )
        if not p:
            return

        img = cv2.imread(p)
        if img is None:
            messagebox.showerror("Blad", "Nie udalo się wczytac obrazu.")
            return

        self._last_frame_bgr = img.copy()
        global detected_objects, selected_id, selected_point
        selected_id = None
        selected_point = None

        try:
            boxes = detect_brick_boxes_yolo(img, conf_thr=0.25)
        except Exception as e:
            print("YOLO error:", repr(e))
            boxes = []

        objs = []
        for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
            roi = crop_roi(img, (x1, y1, x2, y2))
            if roi is None or roi.size == 0:
                continue

            logits = tta_predict(self.interp, self.input_index, self.output_index, roi, size=self.model_input_size, repeats=TTA_REPEATS)
            p_type = softmax(logits)
            _, p_type = apply_low_conf_rules(p_type, self.class_names)
            top_ids = np.argsort(p_type)[-5:][::-1]

            color_name = None
            p_color = None
            if self.color_interp is not None and self.color_class_names:
                x = color_preprocess_bgr(roi, self.color_input_size)
                self.color_interp.set_tensor(self.color_input_index, x)
                self.color_interp.invoke()
                logits_c = self.color_interp.get_tensor(self.color_output_index)[0]
                p_color = softmax(logits_c)
                color_name = self.color_class_names[int(np.argmax(p_color))]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            objs.append({
                "id": i,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "p_det": conf,
                "type_top_ids": top_ids,
                "p_type": p_type,
                "type_name": self.class_names[int(top_ids[0])],
                "type_p": float(p_type[int(top_ids[0])]),
                "color_name": color_name,
                "p_color": p_color
            })

        detected_objects = objs
        active = None
        if not self.var_focus_selected.get():
            active = max(objs, key=lambda o: o["p_det"]) if objs else None
            if active is not None:
                selected_id = active["id"]
        else:
            selected_id = None

        vis = self.draw_overlay(img, fps=0.0, objs=detected_objects, active=active)

        try:
            self.frame_queue.put_nowait((vis, detected_objects, active))
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except Exception:
                pass
            self.frame_queue.put_nowait((vis, detected_objects, active))

        self.status.config(text=f"Wczytano obraz: {os.path.basename(p)} | YOLO: {len(detected_objects)}")

    def draw_overlay(self, frame_bgr, fps: float, objs, active):
        global selected_id, selected_point
        frame = frame_bgr.copy()
        h, w = frame.shape[:2]
        params = overlay_scale_params(h, w)
        panel_h = params["panel_h"]
        thickness = params["thickness"]
        text_scale = params["text_scale"]
        text_thickness = params["text_thickness"]
        focus_mode = bool(self.var_focus_selected.get())

        for o in (objs or []):
            x1, y1, x2, y2 = o["bbox"]
            is_sel = (selected_id == o["id"])
            col = (0, 255, 255) if is_sel else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)

            show_label = True
            if focus_mode and selected_id is None:
                show_label = False
            if focus_mode and selected_id is not None and not is_sel:
                show_label = False

            if show_label:
                label = f"{o['type_name']} {o['type_p']:.2f}"
                if o.get("color_name") is not None:
                    label += f" | {o['color_name']}"
            else:
                label = f"det {o['p_det']:.2f}"

            y_text = max(int(12 * params["s"]), y1 - int(6 * params["s"]))
            cv2.putText(
                frame,
                label,
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                col,
                text_thickness,
                cv2.LINE_AA
            )

        if selected_point is not None:
            sx, sy = selected_point
            cv2.circle(frame, (int(sx), int(sy)), params["click_r1"], (255, 0, 0), -1)
            cv2.circle(frame, (int(sx), int(sy)), params["click_r2"], (255, 255, 255), max(1, int(2 * params["s"])))

        panel = np.full((panel_h, w, 3), 30, dtype=np.uint8)
        x_left = int(12 * params["s"])
        y_base = int(panel_h * 0.62)

        cv2.putText(
            panel,
            f"Obiekty: {len(objs or [])}",
            (x_left, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * params["s"],
            (255, 255, 255),
            max(1, int(2 * params["s"])),
            cv2.LINE_AA
        )

        if selected_id is not None:
            cv2.putText(
                panel,
                f"Wybrany: ID {selected_id}",
                (int(220 * params["s"]), y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * params["s"],
                (255, 255, 0),
                max(1, int(2 * params["s"])),
                cv2.LINE_AA
            )
        else:
            cv2.putText(
                panel,
                "Wybrany: -",
                (int(220 * params["s"]), y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * params["s"],
                (200, 200, 200),
                max(1, int(2 * params["s"])),
                cv2.LINE_AA
            )

        if fps > 0.0:
            cv2.putText(
                panel,
                f"{fps:.1f} FPS",
                (w - int(160 * params["s"]), y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * params["s"],
                (0, 255, 0),
                max(1, int(2 * params["s"])),
                cv2.LINE_AA
            )

        return np.vstack([frame, panel])

    def _clear_bars(self):
        for lbl, bar in self._bars:
            lbl.config(text="—")
            bar["value"] = 0.0

    def _update_bars_for_obj(self, obj):
        p = obj.get("p_type")
        top_ids = obj.get("type_top_ids")
        if p is None or top_ids is None:
            self._clear_bars()
            return

        for i, (lbl, bar) in enumerate(self._bars):
            if i < len(top_ids):
                cid = int(top_ids[i])
                lbl.config(text=self.class_names[cid][:26])
                bar["value"] = float(p[cid])
            else:
                lbl.config(text="—")
                bar["value"] = 0.0

    def _update_side_panel_all(self, objs, active):
        self.result_text.delete("1.0", tk.END)
        if not objs:
            self.result_text.insert(tk.END, "Brak wykrytych klockow.")
            self._clear_bars()
            return

        lines = []
        for o in objs:
            star = "*" if (active is not None and o["id"] == active["id"]) else " "
            lines.append(
                f"{star} ID {o['id']:>2} | det={o['p_det']:.3f} | {o['type_name']} p={o['type_p']:.3f} | {o.get('color_name') or '-'}"
            )
        self.result_text.insert(tk.END, "\n".join(lines))

        if active is not None:
            self._update_bars_for_obj(active)
        else:
            self._clear_bars()

    def _update_side_panel_selected_only(self, objs, active):
        global selected_id
        self.result_text.delete("1.0", tk.END)

        if not objs:
            self.result_text.insert(tk.END, "Brak wykrytych klockow.")
            self._clear_bars()
            return

        if selected_id is None:
            self.result_text.insert(tk.END, "Tryb wyboru: kliknij klocek.\n")
            self.result_text.insert(tk.END, "Wyniki są liczone, tylko ukryte.\n")
            self._clear_bars()
            return

        if active is None:
            self.result_text.insert(tk.END, "Wybrano ID, ale nie znaleziono obiektu.\n")
            self._clear_bars()
            return

        self.result_text.insert(
            tk.END,
            f"Wybrany ID: {active['id']}\n"
            f"Detekcja: {active['p_det']:.3f}\n"
            f"Typ: {active['type_name']}  p={active['type_p']:.4f}\n"
            f"Kolor: {active.get('color_name') or '-'}\n"
        )
        self._update_bars_for_obj(active)

    def _ui_loop(self):
        global detected_objects

        try:
            vis, objs, active = self.frame_queue.get_nowait()
        except queue.Empty:
            vis, objs, active = None, None, None

        if vis is not None:
            detected_objects = objs if objs is not None else []
            self._last_vis_frame = vis.copy()

            disp = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            ih, iw = disp.shape[:2]
            lw = self.preview.winfo_width() or 640
            lh = self.preview.winfo_height() or 480
            scale = min(lw / iw, lh / ih)
            nw, nh = int(iw * scale), int(ih * scale)

            disp = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_LINEAR)
            imgtk = ImageTk.PhotoImage(Image.fromarray(disp))
            self.preview.configure(image=imgtk)
            self._preview_imgtk = imgtk

            if self.var_focus_selected.get():
                self._update_side_panel_selected_only(detected_objects, active)
            else:
                self._update_side_panel_all(detected_objects, active)

        self.after(15, self._ui_loop)

    def _on_close(self):
        try:
            self._stop()
        finally:
            self.destroy()

if __name__ == "__main__":
    app = LegoStudioApp()
    app.mainloop()
