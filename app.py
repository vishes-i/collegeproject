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
        st.header("🧊 Glacier Melting Tracker (No OpenCV)")
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
    before_file = st.file_uploader("Upload BEFORE image", type=['png','jpg','jpeg'], key='before')
with col2:
    after_file = st.file_uploader("Upload AFTER image", type=['png','jpg','jpeg'], key='after')

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
