import os
import json
import gradio as gr
import numpy as np
from PIL import Image

# ============= Your existing imports/functions ==============
from cosq_extr import (
    grey_image_array,
    crop_to_square,
    compress_image_with_codebook,
    implement_channel,
    dequantize_image,
    create_montage,
    compress_image_with_lloydmax_codebook,
    dequantize_image_lloydmax
)

# ------------------------------------------------------------
# Example polya channel parameters for different "noise levels"
# ------------------------------------------------------------
CHANNEL_PRESETS = {
    "Very High Noise (10% flips)":  {"R": 1, "G": 0, "B": 9,   "M": 1},
    "High Noise (5% flips)":       {"R": 5, "G": 0, "B": 95,  "M": 1},
    "Medium Noise (1% flips)":         {"R": 1, "G": 0, "B": 99,  "M": 1},
    "Low Noise (0.5% flips)":  {"R": 5, "G": 0, "B": 995, "M": 1},
    "No Noise (0% flips)":           {"R": 0, "G": 0, "B": 1,   "M": 1},
}

# Typical codebook bit allocations
B_BITS_OPTIONS = [24, 58, 76]

# Possible 'noise_idx' values for the codebook (COSQ only)
CODEBOOK_NOISE_IDX_OPTIONS = [
    ("0% Noise", 0),
    ("10% Noise", 5),
    ("5% Noise", 6),
    ("1% Noise", 7),
    ("0.5% Noise", 8)
]
NOISE_IDX_MAPPING = {label: value for label, value in CODEBOOK_NOISE_IDX_OPTIONS}

CODEBOOK_FILEPATH_COSQ = 'codebooks.json'
CODEBOOK_FILEPATH_LM = 'lloydmax_codebooks.json'

# Load COSQ codebooks once at startup
try:
    with open(CODEBOOK_FILEPATH_COSQ, 'r') as f:
        CODEBOOKS_DICT_COSQ = json.load(f)
except FileNotFoundError:
    print(f"[ERROR] Could not find {CODEBOOK_FILEPATH_COSQ}.")
    CODEBOOKS_DICT_COSQ = None

# Load Lloyd–Max codebooks once at startup
try:
    with open(CODEBOOK_FILEPATH_LM, 'r') as f:
        CODEBOOKS_DICT_LM = json.load(f)
except FileNotFoundError:
    print(f"[ERROR] Could not find {CODEBOOK_FILEPATH_LM}.")
    CODEBOOKS_DICT_LM = None

def list_images_in_folder(folder_path="images"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def compute_psnr(original_array, reconstructed_array):
    mse = np.mean((original_array - reconstructed_array) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)
    return psnr

def process_image(
    image_filename,
    b_bits,
    noise_idx_label,
    channel_preset_key,
    delta_value
):
    """
    1) Load and crop the image to square
    2) COSQ compress + channel + reconstruct
    3) Lloyd–Max compress + channel + reconstruct
    4) Compute PSNR for both
    """
    # Check codebooks
    if not CODEBOOKS_DICT_COSQ or not CODEBOOKS_DICT_LM:
        err_img = Image.new("RGB", (200, 100), color="red")
        return err_img, err_img, err_img, "Error: Missing codebooks.", "Error: Missing codebooks."

    # Load image from folder
    image_path = os.path.join("images", image_filename)
    grey_image = grey_image_array(image_path)
    image_array = crop_to_square(grey_image)
    h, w = image_array.shape

    # Convert user inputs
    b_bits_int = int(b_bits)
    noise_idx = NOISE_IDX_MAPPING[noise_idx_label]
    polya_params = CHANNEL_PRESETS[channel_preset_key]
    R = polya_params["R"]
    G = polya_params["G"]
    B_param = polya_params["B"]
    M = polya_params["M"]
    Del = int(delta_value)

    # ---------------------
    # COSQ pipeline
    # ---------------------
    comp_array_cosq, dcMean_c, dcStd_c, acStd_c = compress_image_with_codebook(
        image_array, CODEBOOKS_DICT_COSQ, b_bits_int, noise_idx, block_size=8
    )
    noisy_cosq = implement_channel(comp_array_cosq, R, G, B_param, M, Del)
    decoded_blocks_cosq = dequantize_image(
        noisy_cosq,
        CODEBOOKS_DICT_COSQ,
        b_bits_int,
        noise_idx,
        block_size=8,
        dcMean=dcMean_c,
        dcStd=dcStd_c,
        acStd=acStd_c
    )
    cosq_reconstructed = create_montage(decoded_blocks_cosq, h, w, 8)

    # ---------------------
    # Lloyd–Max pipeline
    # ---------------------
    comp_array_lm, dcMean_lm, dcStd_lm, acStd_lm = compress_image_with_lloydmax_codebook(
        image_array, CODEBOOKS_DICT_LM, b_bits_int, block_size=8
    )
    noisy_lm = implement_channel(comp_array_lm, R, G, B_param, M, Del)
    decoded_blocks_lm = dequantize_image_lloydmax(
        noisy_lm,
        CODEBOOKS_DICT_LM,
        b_bits_int,
        block_size=8,
        dcMean=dcMean_lm,
        dcStd=dcStd_lm,
        acStd=acStd_lm
    )
    lm_reconstructed = create_montage(decoded_blocks_lm, h, w, 8)

    # Convert to PIL
    original_pil = Image.fromarray(image_array.astype(np.uint8), mode='L')
    cosq_pil = Image.fromarray(cosq_reconstructed.astype(np.uint8), mode='L')
    lm_pil = Image.fromarray(lm_reconstructed.astype(np.uint8), mode='L')

    # Calculate PSNR
    psnr_cosq = compute_psnr(image_array, cosq_reconstructed)
    psnr_lm = compute_psnr(image_array, lm_reconstructed)

    psnr_cosq_str = f"{psnr_cosq:.2f} dB" if psnr_cosq < float('inf') else "∞ (no error)"
    psnr_lm_str = f"{psnr_lm:.2f} dB" if psnr_lm < float('inf') else "∞ (no error)"

    return original_pil, cosq_pil, lm_pil, psnr_cosq_str, psnr_lm_str

def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# COSQ vs. Lloyd–Max Quantization Comparison")

        with gr.Row():
            image_dropdown = gr.Dropdown(
                label="Select Image",
                choices=list_images_in_folder(),
                value=list_images_in_folder()[0] if list_images_in_folder() else None
            )
            with gr.Column():
                b_bits_dropdown = gr.Dropdown(
                    label="Compression Bits Per Pixel",
                    choices=[str(x) for x in B_BITS_OPTIONS],
                    value="76"
                )
                codebook_idx_dropdown = gr.Dropdown(
                    label="COSQ Codebook Noise Index",
                    choices=[f"{label}" for label, _ in CODEBOOK_NOISE_IDX_OPTIONS],
                    value="10% Noise"
                )
                channel_preset_dropdown = gr.Dropdown(
                    label="Channel Noise Preset",
                    choices=list(CHANNEL_PRESETS.keys()),
                    value="Medium Noise (1% flips)"
                )
                delta_dropdown = gr.Dropdown(
                    label="Delta (Del)",
                    choices=["0", "5", "10", "50", "100", "1000"],
                    value="10"
                )
                run_btn = gr.Button("Run Comparison")

        # Arrange images & PSNR in 3 columns so that each image sits above its PSNR
        with gr.Row():
            with gr.Column():
                original_output = gr.Image(label="Original (Grayscale)", type="pil")


            with gr.Column():
                cosq_output = gr.Image(label="COSQ Reconstructed", type="pil")
                cosq_psnr_text = gr.Textbox(label="COSQ PSNR (dB)")

            with gr.Column():
                lm_output = gr.Image(label="Lloyd–Max Reconstructed", type="pil")
                lm_psnr_text = gr.Textbox(label="Lloyd–Max PSNR (dB)")

        run_btn.click(
            fn=process_image,
            inputs=[
                image_dropdown,
                b_bits_dropdown,
                codebook_idx_dropdown,
                channel_preset_dropdown,
                delta_dropdown
            ],
            outputs=[
                original_output,
                cosq_output,
                lm_output,
                cosq_psnr_text,
                lm_psnr_text
            ]
        )

    return demo

def main():
    demo = build_interface()
    demo.launch(share=True)  # or just demo.launch()

if __name__ == "__main__":
    main()
