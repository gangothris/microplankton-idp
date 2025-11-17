# app.py  ---------------------------------------------------------
import streamlit as st
from PIL import Image
import numpy as np
import io, os, random, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from pathlib import Path

st.set_page_config(page_title="Microplankton — Recon → Merge → Detect", layout="wide")

st.title("Microplankton — Reconstruction → Merge (Grid) → Detection")
st.markdown("Upload 1–5 raw holograms → Reconstruct → Merge → Detect")

# --------------------------------------------------------
# Sidebar
# --------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    recon_size = st.selectbox("Reconstruction Size (px)", [128, 256, 320, 512], index=1)
    conf_th = st.slider("YOLO Confidence Threshold", 0.05, 1.0, 0.25, 0.01)

# --------------------------------------------------------
# Model paths
# --------------------------------------------------------
RECON_MODEL_PATH = "best_fista_unet.pt"
YOLO_MODEL_PATH = "best.pt"

# --------------------------------------------------------
# Load grayscale
# --------------------------------------------------------
def load_gray(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    arr = np.array(img).astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
    return arr

# --------------------------------------------------------
# FISTA-UNet model (same as training)
# --------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self,c_in,c_out):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(c_in,c_out,3,padding=1), nn.ReLU(),
            nn.Conv2d(c_out,c_out,3,padding=1), nn.ReLU()
        )
    def forward(self,x): return self.m(x)

class UNetProx(nn.Module):
    def __init__(self,c_in=1,c_out=1,f=32):
        super().__init__()
        self.d1 = DoubleConv(c_in,f); self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(f,2*f);  self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(2*f,4*f);self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(4*f,8*f)

        self.u3 = nn.ConvTranspose2d(8*f,4*f,2,2); self.c3 = DoubleConv(8*f,4*f)
        self.u2 = nn.ConvTranspose2d(4*f,2*f,2,2); self.c2 = DoubleConv(4*f,2*f)
        self.u1 = nn.ConvTranspose2d(2*f,f,2,2);   self.c1 = DoubleConv(2*f,f)

        self.out = nn.Conv2d(f,1,1)

    def forward(self,x):
        d1=self.d1(x)
        d2=self.d2(self.p1(d1))
        d3=self.d3(self.p2(d2))
        b =self.b(self.p3(d3))
        u3=self.c3(torch.cat([self.u3(b),d3],1))
        u2=self.c2(torch.cat([self.u2(u3),d2],1))
        u1=self.c1(torch.cat([self.u1(u2),d1],1))
        return self.out(u1)

class FISTA_UNet(nn.Module):
    def __init__(self,K=12,f=32):
        super().__init__()
        self.K=K
        self.prox=UNetProx(1,1,f)
        self.s_k=nn.Parameter(torch.zeros(K))
        self.m_k=nn.Parameter(torch.zeros(K))

    def forward(self,y):
        x_prev=y; y_cur=y
        for k in range(self.K):
            alpha=F.softplus(self.s_k[k])+1e-3
            beta =0.9*torch.tanh(self.m_k[k])
            grad =(y_cur-y)
            r    =y_cur-alpha*grad
            x_k  =r+self.prox(r)
            y_cur=x_k if k==0 else x_k+beta*(x_k-x_prev)
            x_prev=x_k
        return x_k

# --------------------------------------------------------
# Load Models
# --------------------------------------------------------
@st.cache_resource
def load_recon_model():
    if not os.path.exists(RECON_MODEL_PATH):
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = FISTA_UNet().to(device)
    m.load_state_dict(torch.load(RECON_MODEL_PATH, map_location=device))
    m.eval()
    return m

@st.cache_resource
def load_yolo():
    if not os.path.exists(YOLO_MODEL_PATH):
        return None
    return YOLO(YOLO_MODEL_PATH)

recon_model = load_recon_model()
yolo_model = load_yolo()

if recon_model is None:
    st.error("Reconstruction model missing.")
if yolo_model is None:
    st.error("YOLO model missing.")

# --------------------------------------------------------
# Reconstruction
# --------------------------------------------------------
def reconstruct(arr, model, size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inp = cv2.resize((arr*255).astype(np.uint8), (size, size)) / 255.0
    t = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(t).clamp(0,1).cpu().numpy()[0,0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out

# --------------------------------------------------------
# NEW GRID-BASED MERGE (No Overlap, All Images appear)
# --------------------------------------------------------
def merge_grid(instances, canvas_size=(640,640)):
    H, W = canvas_size
    canvas = np.zeros((H,W), dtype=np.uint8)

    n = len(instances)
    grid_cols = int(np.ceil(np.sqrt(n)))
    grid_rows = int(np.ceil(n / grid_cols))

    cell_w = W // grid_cols
    cell_h = H // grid_rows

    anns = []
    idx = 0

    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= n:
                break
            inst = instances[idx]
            ih, iw = inst.shape

            scale = min((cell_w * 0.8) / iw, (cell_h * 0.8) / ih)
            new_w = int(iw * scale)
            new_h = int(ih * scale)

            resized = cv2.resize((inst*255).astype(np.uint8), (new_w,new_h))

            x0 = c*cell_w + (cell_w - new_w)//2
            y0 = r*cell_h + (cell_h - new_h)//2

            canvas[y0:y0+new_h, x0:x0+new_w] = resized
            anns.append({"bbox":[x0,y0,new_w,new_h], "id":idx})

            idx += 1

    return canvas, anns

# --------------------------------------------------------
# Upload UI
# --------------------------------------------------------
uploaded = st.file_uploader("Upload 1–5 raw hologram images", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploaded:
    if len(uploaded) > 5:
        st.error("Upload max 5 images only.")
        st.stop()

    raw_arrays = [load_gray(u) for u in uploaded]
    filenames  = [u.name for u in uploaded]

    if st.button("Run Reconstruction"):
        recons = [reconstruct(arr, recon_model, recon_size) for arr in raw_arrays]
        st.session_state["recons"] = recons
        st.session_state["names"] = filenames

        st.header("Raw & Reconstructed")
        for name, raw, rec in zip(filenames, raw_arrays, recons):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(name)
                st.image((raw*255).astype(np.uint8), caption="Raw")
            with c2:
                st.subheader("Reconstructed")
                st.image((rec*255).astype(np.uint8), caption=f"{recon_size}px")

        st.success("Reconstruction complete. Now you can Merge & Detect.")

# --------------------------------------------------------
# Merge + Detection
# --------------------------------------------------------
if "recons" in st.session_state:

    st.markdown("---")
    st.header("Merge & Detection")

    if st.button("Run Merge"):
        merged, anns = merge_grid(st.session_state["recons"])
        st.session_state["merged"] = merged

        st.subheader("Merged Image")
        st.image(merged, caption="Grid Merge (all objects placed)")

        bio = io.BytesIO()
        Image.fromarray(merged).save(bio, format="PNG")
        st.download_button("Download Merged", bio.getvalue(), "merged.png")

    if "merged" in st.session_state:
        if st.button("Run YOLO Detection"):
            img = st.session_state["merged"]
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            results = yolo_model.predict(bgr, conf=conf_th, imgsz=640, verbose=False)
            r = results[0]

            boxes = []
            labels = []

            if r.boxes is not None:
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = yolo_model.model.names.get(cls, f"class_{cls}")
                    boxes.append((x1,y1,x2,y2))
                    labels.append(f"{name} {conf:.2f}")

            draw = bgr.copy()
            for (x1,y1,x2,y2), label in zip(boxes, labels):
                cv2.rectangle(draw, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(draw, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            st.subheader("Detections")
            st.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))

            bio = io.BytesIO()
            _, buf = cv2.imencode(".png", draw)
            st.download_button("Download Detection", buf.tobytes(), "detection.png")

