import streamlit as st
from util import DataLoader
from PIL import Image
import numpy as np

# Tạo loader
loader = DataLoader(retri_dir="result_clip")

# Giao diện Streamlit
st.title("Sample Retrieval Viewer")

# Nhập sample_id
sample_id = st.text_input("Enter Sample ID:")

if sample_id:
    try:
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_retri_data(sample_id)
        
        st.subheader("📌 Question")
        st.write(question)
        
        st.subheader("✅ Answer")
        st.write(answer)
        
        st.subheader("🔎 Query")
        st.write(query)

        st.subheader("🖼️ Retrieved Images")
        for idx, img in enumerate(retri_imgs):
            st.image(Image.fromarray(img) if isinstance(img, np.ndarray) else img, caption=f"Image {idx+1}", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error loading sample ID {sample_id}: {e}")
