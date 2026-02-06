import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import io
from PIL import Image

# ────────────────────────────────────────────────
# Load the model (cached to run only once)
# ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Stable Diffusion model... may take a few minutes")
def load_pipeline(model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,  # disable safety checker (optional)
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe

# ────────────────────────────────────────────────
# Page configuration
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖼️ AI Image Generator")
st.markdown(
    "**Type a description of the image you imagine** and AI will generate it in seconds using Stable Diffusion."
)

# ────────────────────────────────────────────────
# Sidebar - Settings
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selection
    st.markdown("Using pre-selected lightweight model: `v1-5` for fast and easy usage")
    model_id = "runwayml/stable-diffusion-v1-5"

    st.markdown("---")

    # Number of steps
    num_steps = st.slider(
        "Number of inference steps",
        15, 50, 25, 5,
        help="Higher number = better quality but takes longer"
    )

    # Guidance Scale (CFG)
    guidance_scale = st.slider(
        "Guidance scale",
        1.0, 12.0, 7.5, 0.5,
        help="Values around 7–9 usually give best balance"
    )

    # Image dimensions
    st.subheader("Image dimensions")
    col1, col2 = st.columns(2)
    with col1:
        width = st.select_slider("Width", [512, 640, 768], value=512)
    with col2:
        height = st.select_slider("Height", [512, 640, 768], value=512)

    # Seed
    seed = st.number_input(
        "Random seed",
        value=42, step=1,
        help="Same number = same result with same settings"
    )

    st.markdown("---")
    generate_button = st.button("✨ Generate Image", type="primary", use_container_width=True)

# ────────────────────────────────────────────────
# Main input area
# ────────────────────────────────────────────────
col_prompt, col_neg = st.columns([3, 2])

with col_prompt:
    prompt = st.text_area(
        "**Type your image description**",
        value="Futuristic city at night, blue and pink neon lights, light rain, cyberpunk atmosphere, high detail, cinematic",
        height=140,
        key="prompt_input"
    )

with col_neg:
    negative_prompt = st.text_area(
        "**Negative prompt (optional)**",
        value="blurry, low quality, distorted, ugly, extra limbs, wrong anatomy, watermark, text, logo",
        height=140,
        key="negative_prompt"
    )

# ────────────────────────────────────────────────
# Generate image
# ────────────────────────────────────────────────
if generate_button and prompt.strip():
    with st.spinner("Generating image... may take 10–40 seconds depending on settings"):
        try:
            pipe = load_pipeline(model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(int(seed))

            with torch.autocast(device):
                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt.strip() else None,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                ).images[0]

            st.success("Image generated successfully!")
            st.image(image, caption=prompt, use_column_width=True)

            # Download button
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ Download Image (PNG)",
                data=buffered.getvalue(),
                file_name="ai_generated_image.png",
                mime="image/png",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error occurred while generating image: {str(e)}")
            st.info("Common solutions:\n• Reduce image dimensions\n• Reduce number of steps\n• Make sure there is enough GPU memory")

# ────────────────────────────────────────────────
# Tips / info
# ────────────────────────────────────────────────
with st.expander("Tips & Important Info"):
    st.markdown("""
    - First run requires downloading the model (~2–4 GB for v1-5).
    - Use smaller dimensions (512x512) if GPU memory is limited.
    - Writing the prompt in English usually gives better results, but Arabic is supported as well.
    """)

st.caption("Built with Streamlit + Hugging Face Diffusers • 2026")
