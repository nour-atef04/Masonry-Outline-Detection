import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import binary_dilation
from skimage.morphology import h_minima
from skimage.segmentation import watershed, mark_boundaries, find_boundaries
import matplotlib.pyplot as plt
import ezdxf
import os
from scipy import ndimage as ndi
import tifffile as tiff
from streamlit_drawable_canvas import st_canvas
import math
from streamlit_image_zoom import image_zoom
import torch
import sys
import os

print("✅ streamlit_app.py is running")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Standard U-Net architecture with encoder-decoder for image segmentation
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Encoder path
        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final output layer (1-channel probability map)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    # Forward pass through the U-net    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.final(d1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
# model.load_state_dict(torch.load("best-model-5/kaggle/working/best_unet_bricks.pth", map_location=device))
model_path = resource_path("model.pth")
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model = UNet().to(device)
model.load_state_dict(state_dict)
model.eval()

# Image transforms
transform = A.Compose([
    A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(min_height=None, min_width=None, 
                  pad_height_divisor=16, pad_width_divisor=16, border_mode=0),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(image, gamma=1.5, clahe_clip=2.0, use_bilateral=False, bilateral_d=9, 
                     bilateral_sigmaColor=75, bilateral_sigmaSpace=75):
    
    # Convert to LAB and apply CLAHE to L channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(3,3))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0,256)]).astype("uint8")
    image = cv2.LUT(image, table)

    # Optional smoothing using bilateral fitering
    if use_bilateral:
        image = cv2.bilateralFilter(image, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
    return image

# Run image though the U-net to get brick probability map
def predict(image_np):
    transformed = transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).cpu().numpy()[0,0]
    return prob_map

# Apply threshold, distance transform, h-minima and watershed segmentation
def postprocess(prob_map, image_np, h=15, threshold=0.5):
    binary_mask = (prob_map>threshold).astype(np.uint8)
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    inv_distance = np.max(distance)-distance
    inv_distance = cv2.GaussianBlur(inv_distance,(5,5),0).astype(np.float32)

    # if very large, downscale for watershed
    from skimage.transform import resize
    if prob_map.shape[0] > 3000 or prob_map.shape[1] > 3000:
        scale = 0.05 
        prob_map = resize(prob_map, (int(prob_map.shape[0]*scale), int(prob_map.shape[1]*scale)), preserve_range=True, anti_aliasing=True)
        image_np = resize(image_np, (int(image_np.shape[0]*scale), int(image_np.shape[1]*scale)), preserve_range=True, anti_aliasing=True).astype(np.uint8)

    hminima_map = h_minima(inv_distance, h=h)
    markers, _ = ndi.label(hminima_map)
    labels = watershed(inv_distance, markers=markers, mask=binary_mask)
    boundaries = find_boundaries(labels, mode='outer')
    return labels, boundaries

# Extract contours and save as DXF
def save_dxf(labels, path="outlines.dxf", epsilon=1.0):
    unique_labels = np.unique(labels)
    doc = ezdxf.new()
    msp = doc.modelspace()
    height = labels.shape[0]

    for label in unique_labels:
        if label == 0:  # skip background
            continue
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            if epsilon > 0:
                approx = cv2.approxPolyDP(c, epsilon, True)
            else:
                approx = c
            points = [(float(p[0][0]), float(height - p[0][1])) for p in approx]
            if len(points) > 1:
                msp.add_lwpolyline(points, close=True)

    doc.saveas(path)
    return path

# Process large images by splitting into overlapping tiles and blend outputs
def process_large_image(image, gamma=1.5, clahe_clip=2.0, threshold=0.5, h_val=10, base_tile_size=2048):
    """Process large image using adaptive tiling"""

    H, W = image.shape[:2]

    # Adjust tile size based on image size
    if max(H, W) > 8192:  
        base_tile_size = 4096
    elif max(H, W) > 4096: 
        base_tile_size = 3072
    else:  
        base_tile_size = 2048

    tile_size = (base_tile_size // 16) * 16
    overlap = max(64, tile_size // 7)  
    
    # Initilize full probability map and weight counts
    full_prob = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)  
    
    # Compute number of tiles
    x_steps = math.ceil((W - overlap) / (tile_size - overlap))
    y_steps = math.ceil((H - overlap) / (tile_size - overlap))
    total_tiles = x_steps * y_steps
    processed_tiles = 0

    progress_bar = st.progress(0)
    
    for y in range(0, H, tile_size - overlap):
        for x in range(0, W, tile_size - overlap):
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(W, x + tile_size)
            y2 = min(H, y + tile_size)
            
            tile = image[y1:y2, x1:x2]
            
            if tile.size == 0:
                continue
                
            pre_tile = preprocess_image(tile, gamma=gamma, clahe_clip=clahe_clip)
            prob_tile = predict(pre_tile)
            
            prob_tile_resized = cv2.resize(prob_tile, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
            
            if overlap > 0:

                # Blending -> apply smooth masks on overlap areas
                mask = np.ones((y2-y1, x2-x1), dtype=np.float32)
                
                if x > 0:
                    mask[:, :overlap] = np.linspace(0, 1, overlap)
                
                if x + tile_size < W:
                    mask[:, -overlap:] = np.linspace(1, 0, overlap)
                
                if y > 0:
                    mask[:overlap, :] *= np.linspace(0, 1, overlap)[:, np.newaxis]
                
                if y + tile_size < H:
                    mask[-overlap:, :] *= np.linspace(1, 0, overlap)[:, np.newaxis]
                
                full_prob[y1:y2, x1:x2] += prob_tile_resized * mask
                counts[y1:y2, x1:x2] += mask
            else:
                full_prob[y1:y2, x1:x2] += prob_tile_resized
                counts[y1:y2, x1:x2] += 1
            
            processed_tiles += 1
            progress_bar.progress(processed_tiles / total_tiles)
    
    # Normalize by counts to blend overlaps smoothly
    counts[counts == 0] = 1
    full_prob /= counts
    
    return full_prob

# ----------------------------------------------------
# --------------- Streamlit interface ----------------
# ----------------------------------------------------

st.title("Brick Segmentation App")

uploaded = st.file_uploader("Upload an image", type=['png','jpg','jpeg','tif','tiff'])
gamma = st.slider("Gamma correction",1.0,3.0,1.5,0.1, help="Adjust brightness/contrast")
clahe_clip = st.slider("CLAHE clip limit",1.0,5.0,2.0,0.1, help="Higher = more local contrast but may introduce noise")
with st.expander("Optional: Bilateral Filtering (noise smoothing)"):
    use_bilateral = st.checkbox("Enable bilateral filtering", value=True)
    bilateral_d = st.slider("Diameter (d)", 1, 15, 9, 2, help="Larger = smoother")
    bilateral_sigmaColor = st.slider("Sigma Color", 1, 150, 75, 5, help="Higher = more smoothing across colors")
    bilateral_sigmaSpace = st.slider("Sigma Space", 1, 150, 75, 5, help="Higher = smoother over larger areas")
threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01, help="Lower = detects more regions (more sensitive); higher = keeps only regions with higher model confidence")
h_val = st.slider("Watershed h-minima (controls split)",1,20,10,1, help="Higher = fewer splits (merge small regions); lower = more splits")

if uploaded:
    filename = uploaded.name.lower()
    if filename.endswith(('.tif', '.tiff')):
        with tiff.TiffFile(uploaded) as tif:
            image = tif.pages[0].asarray()
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
    else:
        image = np.array(Image.open(uploaded).convert('RGB'))

    st.write("## Select preview region to test parameters")

    preview_scale = 512 / max(image.shape[0], image.shape[1])
    small_image = cv2.resize(image, (int(image.shape[1]*preview_scale), int(image.shape[0]*preview_scale)))
    small_image_pil = Image.fromarray(small_image)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=small_image_pil,
        update_streamlit=True,
        height=small_image.shape[0],
        width=small_image.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    crop_x = crop_y = 0 
    crop_w = crop_h = 512

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        left = int(obj["left"])
        top = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])

        crop_x = int(left / preview_scale)
        crop_y = int(top / preview_scale)
        crop_w = int(width / preview_scale)
        crop_h = int(height / preview_scale)

        crop_x = min(crop_x, image.shape[1]-1)
        crop_y = min(crop_y, image.shape[0]-1)
        crop_w = min(crop_w, image.shape[1]-crop_x)
        crop_h = min(crop_h, image.shape[0]-crop_y)

    preview = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

    pre = preprocess_image(preview, gamma=gamma, clahe_clip=clahe_clip, use_bilateral=use_bilateral, 
    bilateral_d=bilateral_d, 
    bilateral_sigmaColor=bilateral_sigmaColor, 
    bilateral_sigmaSpace=bilateral_sigmaSpace)
    prob = predict(pre)
    labels, boundaries = postprocess(prob, pre, h=h_val, threshold=threshold)

    st.subheader("Preview")
    st.image(pre, caption="Preprocessed preview")
    boundaries_uint8 = (boundaries * 255).astype(np.uint8)
    st.image(boundaries_uint8, caption="Boundaries (preview)")

    preview_rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB) if preview.ndim==2 or preview.shape[2]==1 else preview.copy()
    overlay_preview = preview_rgb.copy()
    boundaries_resized = cv2.resize(boundaries.astype(np.uint8), 
                                (overlay_preview.shape[1], overlay_preview.shape[0]), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay_preview[boundaries_resized] = [255, 0, 0]

    # st.image(overlay_preview, caption="Preview with Watershed Boundaries")
    overlay_pil = Image.fromarray(overlay_preview)
    st.subheader("Zoomable preview with boundaries")
    image_zoom(overlay_pil)


    dxf_smoothness = st.slider(
    "DXF smoothness", 0.0, 5.0, 0.5, 0.1, key="preview_smoothness", help="Lower = preserves detailed shapes; higher = simplifies to fewer points"
)
    
    height, width = labels.shape
    dxf_mask = np.zeros((height, width, 3), dtype=np.uint8) 

    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            if dxf_smoothness > 0:
                approx = cv2.approxPolyDP(c, epsilon=dxf_smoothness, closed=True)
            else:
                approx = c
            cv2.polylines(dxf_mask, [approx], isClosed=True, color=(255, 255, 255), thickness=1)

    st.image(dxf_mask, caption=f"Preview of DXF polygons only (black background, epsilon={dxf_smoothness})")



    if st.button("Export DXF (preview region)"):
        # preview_dxf_path = save_dxf(boundaries, path="preview_outlines.dxf", epsilon=dxf_smoothness)
        preview_dxf_path = save_dxf(labels, path="preview_outlines.dxf", epsilon=dxf_smoothness)

        st.success(f"Preview DXF saved: {os.path.abspath(preview_dxf_path)}")
        with open(preview_dxf_path, "rb") as f:
            st.download_button("Download Preview DXF", f, file_name="preview_outlines.dxf")

    if st.button("Run on full image"):
        with st.spinner("Processing full image (this may take a while for large images)..."):

            base_tile_size = 2048  

            if crop_w > 0 and crop_h > 0:
                max_preview_dim = max(crop_w, crop_h)
                base_tile_size = int(min(max_preview_dim * 1.5, 4096))  
                base_tile_size = (base_tile_size // 16) * 16
            else:
                base_tile_size = 2048


            st.info(f"Using tile size: {base_tile_size}px for full image processing.")

            full_prob = process_large_image(image, gamma=gamma, clahe_clip=clahe_clip, 
                                threshold=threshold, h_val=h_val, base_tile_size=base_tile_size)

            labels_full, boundaries_full = postprocess(full_prob, image, h=h_val, threshold=threshold)

            overlay_full = image.copy()
            diag = int((image.shape[0]**2 + image.shape[1]**2) ** 0.5)
            thickness = max(1, diag // 1000) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
            boundary_full_uint8 = (boundaries_full * 255).astype(np.uint8)
            thick_boundary_full = cv2.dilate(boundary_full_uint8, kernel)
            overlay_full[thick_boundary_full > 0] = [255,0,0]

            st.session_state["boundaries_full"] = boundaries_full
            st.session_state["labels_full"] = labels_full
            st.session_state["overlay_full"] = overlay_full

    if "overlay_full" in st.session_state:
        st.subheader("Full Image Results")
        boundaries_uint8 = (st.session_state["boundaries_full"] * 255).astype(np.uint8)
        st.image(boundaries_uint8, caption="Watershed boundaries (full image)")
        # st.image(st.session_state["overlay_full"], caption="Original Image with Watershed Boundaries")
        full_overlay_pil = Image.fromarray(st.session_state["overlay_full"])
        st.subheader("Zoomable full image with boundaries")
        image_zoom(full_overlay_pil)

        dxf_smoothness_full = st.slider(
            "DXF smoothness for the full image", 
            0.0, 5.0, 0.5, 0.1, key="full_smoothness",
            help="Lower = preserves detailed shapes; higher = simplifies to fewer points"
        )

        height, width = st.session_state["labels_full"].shape
        dxf_mask_full = np.zeros((height, width, 3), dtype=np.uint8)

        unique_labels_full = np.unique(st.session_state["labels_full"])
        for label in unique_labels_full:
            if label == 0:
                continue
            mask = (st.session_state["labels_full"] == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                if dxf_smoothness_full > 0:
                    approx = cv2.approxPolyDP(c, epsilon=dxf_smoothness_full, closed=True)
                else:
                    approx = c
                cv2.polylines(dxf_mask_full, [approx], isClosed=True, color=(255, 255, 255), thickness=1)

        st.image(dxf_mask_full, caption=f"DXF polygons preview (black background, epsilon={dxf_smoothness_full})")


        if st.button("Export DXF (full image)"):
            path = save_dxf(st.session_state["labels_full"], epsilon=dxf_smoothness_full)
            st.success(f"DXF saved: {os.path.abspath(path)}")
            with open(path, "rb") as f:
                st.download_button("Download DXF", f, file_name="brick_outlines.dxf")
