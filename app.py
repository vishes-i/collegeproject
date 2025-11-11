# Glacier Melting — Portfolio Streamlit App (No OpenCV)
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import io, base64, tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Glacier Portfolio & CV", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- Styles ----------
st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0f172a,#0ea5e9); padding:30px; border-radius:12px; color:white}
    .sub {color: #e2e8f0}
    .card {background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:18px; border-radius:12px; box-shadow: 0 8px 20px rgba(2,6,23,0.4)}
    .sm {font-size:0.9rem; color:#cbd5e1}
      .sub{color:green;font-size:large}
    </style>
    """, unsafe_allow_html=True)

# ---------- Header ----------
with st.container():
    left, right = st.columns([3, 1])
    with left:
        st.header("🧊 Glacier Melting Tracker ")
        st.markdown('<div class="header">', unsafe_allow_html=True)
        st.title("Vishesh Kumar Prajapati — Computer Vision & Data Science")
        st.markdown("<p  class='sub'>Portfolio · Computer Vision · Python · Streamlit</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.image("https://placekitten.com/200/200", caption="SHIVSHAKTI", width=200)

st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.title("Tracker Controls")
colormap = st.sidebar.selectbox("Diff colormap", options=["viridis", "plasma", "inferno", "magma"], index=0)
thresholding = st.sidebar.slider("Diff threshold (for mask)", 1, 255, 30)

# ---------- Utility ----------
def resize_pil(img_pil, max_dim=800):
    w, h = img_pil.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        return img_pil.resize((int(w*scale), int(h*scale)))
    return img_pil

def compute_diff_pil(img1, img2, thresh=30):
    arr1 = np.array(img1.convert("L"))
    arr2 = np.array(img2.convert("L"))
    d = np.abs(arr1.astype("int16") - arr2.astype("int16")).astype("uint8")
    mask = (d > thresh).astype(np.uint8) * 255
    return d, mask

def apply_colormap(diff_gray, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    normed = diff_gray / 255.0
    colored = (cmap(normed)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)

# ---------- Main ----------
st.header("Glacier Melting Tracker")
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Upload BEFORE image", type=['png','jpg','jpeg'], key='before_main')
with col2:
    after_file = st.file_uploader("Upload AFTER image", type=['png','jpg','jpeg'], key='after_main')


if before_file and after_file:
    before_img = resize_pil(Image.open(before_file))
    after_img = resize_pil(Image.open(after_file))

    st.image(before_img, caption="BEFORE")
    st.image(after_img, caption="AFTER")

    diff_gray, mask = compute_diff_pil(before_img, after_img, thresh=thresholding)
    heat = apply_colormap(diff_gray, cmap_name=colormap)

    st.image(heat, caption="Difference Heatmap")
    st.image(mask, caption="Binary Mask")

    pct = (np.count_nonzero(mask) / mask.size) * 100
    st.metric("Percent changed (approx)", f"{pct:.2f}%")

else:

    st.info("Upload BEFORE and AFTER images to compare.")

st.header("Glacier Melting Tracker")
st.write(
    "Upload a *BEFORE* and *AFTER* image (satellite / drone / photo). The app will align, compute differences and show an interactive before/after comparison.")


col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("Upload BEFORE image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='before')
with col2:
    after_file = st.file_uploader("Upload AFTER image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='after')

use_sample = False
if not before_file or not after_file:
    st.info("You can try sample glacier images if you don't have your own. (Small images for demo.)")
    if st.button("Load sample images"):
        use_sample = True
        # Load from package - create two synthetic demo images
        demo_dir = tempfile.gettempdir()
        b = Image.new('RGB', (900, 600), (150, 180, 220))
        a = Image.new('RGB', (900, 600), (150, 180, 220))
        # Draw a white 'ice' blob that shrinks in AFTER
        cvb = pil_to_cv(b)
        cva = pil_to_cv(a)
        cv2.circle(cvb, (450, 300), 200, (240, 240, 255), -1)
        cv2.circle(cva, (450, 300), 150, (240, 240, 255), -1)
        before_img = cvb
        after_img = cva
    else:
        before_img = None
        after_img = None
else:
    before_img = pil_to_cv(read_image(before_file))
    after_img = pil_to_cv(read_image(after_file))

if before_img is not None and after_img is not None:
    # Resize to reasonable size
    before_img = resize_keeping_aspect(before_img, max_dim=1200)
    after_img = resize_keeping_aspect(after_img, max_dim=1200)

    st.markdown("### Alignment & Difference")
    colA, colB = st.columns([1, 1])
    with colA:
        st.image(cv_to_pil(before_img), caption='BEFORE (aligned to original orientation)')
    with colB:
        st.image(cv_to_pil(after_img), caption='AFTER (raw upload)')

    # Align
    if alignment_algo == "ORB+Homography":
        aligned_after, H = align_orb(before_img, after_img)
    else:
        aligned_after, H = align_orb(before_img, after_img)

    if H is None:
        st.warning(
            "Could not compute a robust homography — showing unaligned AFTER image. Try images with overlapping regions or different algorithm settings.")
        aligned_after = after_img
    else:
        st.success("Alignment succeeded. Homography matrix computed.")

    st.write("**Homography (first 3x3 block):**")
    st.code(np.array_str(H, precision=3)) if H is not None else None

    st.markdown("---")
    diff_gray, mask = compute_diff(before_img, aligned_after, thresh=thresholding)
    heat = apply_colormap(diff_gray, cmap_name=colormap)

    # Optionally refine mask with segmentation
    if use_segmentation and seg_model is not None:
        try:
            seg_mask_before = segment_image_pil(cv_to_pil(before_img))
            seg_mask_after = segment_image_pil(cv_to_pil(aligned_after))
            if seg_mask_before is not None and seg_mask_after is not None:
                seg_mask = cv2.bitwise_and(seg_mask_before, seg_mask_after)
                mask = cv2.bitwise_and(mask, seg_mask)
                st.sidebar.info("Refined diff mask using DeepLabV3 segmentation.")
        except Exception as e:
            st.sidebar.warning("Segmentation refinement failed: {}".format(e))

    # Overlay heat on top of BEFORE image for visualization
    overlay = cv2.addWeighted(before_img, 0.6, heat, 0.4, 0)
    overlay_pil = cv_to_pil(overlay)
    heat_pil = cv_to_pil(heat)

    # Display interactive before-after slider via simple HTML
    st.markdown("**Interactive comparison (drag):**")


    # Prepare images as base64
    def pil_to_b64(img_pil):
        buffered = io.BytesIO()
        img_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


    b64_before = pil_to_b64(cv_to_pil(before_img))
    b64_after = pil_to_b64(overlay_pil)

    slider_html = f"""
    <style>
    .comp-wrap {{width:100%; max-width:1100px; margin:auto}}
    .comp-img {{width:100%; display:block}}
    .comp-slider {{-webkit-appearance: none; width:100%;}}
    </style>
    <div class="comp-wrap">
      <div style="position:relative;">
        <img src="data:image/png;base64,{b64_before}" class="comp-img" id="img1">
        <img src="data:image/png;base64,{b64_after}" class="comp-img" id="img2" style="position:absolute; top:0; left:0; clip:rect(0px,600px,9999px,0px);">
      </div>
      <input type="range" min="0" max="100" value="50" id="s" class="comp-slider">
    </div>
    <script>
    const s = document.getElementById('s');
    const img2 = document.getElementById('img2');
    s.oninput = function(){{
      const val = this.value/100.0;
      const w = img2.naturalWidth;
      const clipx = Math.round(w * val);
      img2.style.clip = 'rect(0px,'+clipx+'px,9999px,0px)';
    }}
    </script>
    """

    st.components.v1.html(slider_html, height=520)

    st.markdown("---")
    st.subheader("Diff Mask & Statistics")
    st.image(cv_to_pil(mask), caption='Binary diff mask (areas of change)')
    # Statistics: percentage changed
    pct = (np.count_nonzero(mask) / mask.size) * 100.0
    st.metric("Percent changed (approx)", f"{pct:.4f}%")


    # Download result as ZIP with images & mask
    def make_download_zip():
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for name, img in [('before.png', cv_to_pil(before_img)), ('after_aligned.png', cv_to_pil(aligned_after)),
                              ('overlay.png', overlay_pil), ('mask.png', cv_to_pil(mask))]:
                b = io.BytesIO()
                img.save(b, format='PNG')
                z.writestr(name, b.getvalue())
        return buf.getvalue()


    zipped = make_download_zip()
    st.download_button("Download results (zip)", data=zipped,
                       file_name=f"glacier_results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")

else:
    st.info("Upload both BEFORE and AFTER images (or load sample) to run the tracker.")
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

frames = np.arange(1, 21)
np.random.seed(42)
initial_area = 1000
melting = np.cumsum(np.random.uniform(5, 20, size=20))
areas = np.clip(initial_area - melting, 0, None)

fig, ax = plt.subplots()
ax.plot(frames, areas, marker='o', color='blue')
ax.set_title("Simulated Glacier Melting Tracking")
ax.set_xlabel("Frame / Time Step")
ax.set_ylabel("Glacier Area (arbitrary units)")
ax.grid(True)

st.pyplot(fig)

# ---------- Extra Charts ----------
frames = np.arange(1, 21)
np.random.seed(42)
initial_area = 1000
melting = np.cumsum(np.random.uniform(5, 20, size=20))
areas = np.clip(initial_area - melting, 0, None)

fig, ax = plt.subplots()
ax.plot(frames, areas, marker='o', color='blue')
ax.set_title("Simulated Glacier Melting Tracking")
st.pyplot(fig)

years = np.arange(2000, 2021)
np.random.seed(123)
initial_value = 1000
annual_change = np.random.uniform(-50, -10, size=len(years))
values = np.clip(initial_value + np.cumsum(annual_change), 0, None)
df = pd.DataFrame({"Year": years, "Value": values})
fig, ax = plt.subplots()
ax.plot(df["Year"], df["Value"], marker='o', linestyle='-', color='teal')
st.pyplot(fig)

# ---------- Map ----------
data = {
    "Name": ["Glacier A", "Glacier B", "Glacier C"],
    "Latitude": [61.5, 46.8, 78.9],
    "Longitude": [-149.9, 11.2, 16.0],
    "Size_km2": [120, 80, 200]
}
df = pd.DataFrame(data)
fig = px.scatter_geo(df, lat="Latitude", lon="Longitude", hover_name="Name", size="Size_km2", projection="natural earth")
st.plotly_chart(fig)

# ---------- Team ----------
st.markdown("---")
st.title("Meet Our Team")
teammates = [
    {"name": "Vishesh Kumar Prajapati", "role": "Full Stack Developer", "bio": "Expert in computer vision."},
    {"name": "Sumit Yadav", "role": "B.Tech CSE", "bio": "Specializes in web apps."},
    {"name": "Vijay Kharwar", "role": "B.Tech CSE", "bio": "Keeps project on track."}
]
for member in teammates:
    st.markdown(f"### {member['name']}  \n**{member['role']}**  \n{member['bio']}")
    st.write("---")
