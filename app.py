import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="AI Image Generator", layout="wide")
st.title("🖼️ AI Image Generator (CPU-friendly)")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_option = st.sidebar.selectbox(
    "Select Model",
    ["Stable Diffusion 1.5 (Lightweight, CPU-friendly)"]
)

width = st.sidebar.slider("Width", min_value=128, max_value=768, value=512, step=64)
height = st.sidebar.slider("Height", min_value=128, max_value=768, value=512, step=64)
steps = st.sidebar.slider("Number of inference steps", min_value=5, max_value=50, value=25)
guidance = st.sidebar.slider("Guidance scale", min_value=1.0, max_value=15.0, value=7.5)
seed = st.sidebar.number_input("Random seed (0 = random)", min_value=0, value=42)

# Text inputs
prompt = st.text_area("Type your image description", height=100)
negative_prompt = st.text_area(
    "Negative prompt (optional)",
    value="blurry, low quality, distorted, ugly, extra limbs, wrong anatomy, watermark, text, logo",
    height=80
)

# Button to generate
if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt!")
    else:
        st.info("Generating image... may take a few minutes on CPU")
        
        # Set random seed
        generator = None
        if seed != 0:
            generator = torch.manual_seed(seed)
        
        # Load model (CPU only)
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None
        )
        pipe = pipe.to("cpu")  # Force CPU usage

        # Generate image
        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps
        ).images[0]

        st.image(image, caption="Generated Image", use_column_width=True)
