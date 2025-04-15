import os
import json
import gradio as gr
import numpy as np
from PIL import Image

# Shared imports
from cosq_extr import (
    grey_image_array as cosq_grey_image_array,
    crop_to_square,
    compress_image_with_codebook,
    implement_channel as cosq_implement_channel,
    dequantize_image,
    create_montage,
    compress_image_with_lloydmax_codebook,
    dequantize_image_lloydmax,
    compute_psnr as cosq_compute_psnr
)

from covq_extr import (
    grey_image_array as covq_grey_image_array,
    bitstream_to_indices,
    encode_image_blocks,
    indices_to_bitstream,
    implement_channel as covq_implement_channel,
    decode_image_blocks,
    psnr as covq_psnr
)

# Configuration
NOISE_LEVELS = {
    "10% Noise": {"R": 100, "G": 0, "B": 900, "M": 1, "cosq_idx": 5, "epsilon_idx": 4},
    "5% Noise": {"R": 50, "G": 0, "B": 950, "M": 1, "cosq_idx": 6, "epsilon_idx": 3},
    "1% Noise": {"R": 10, "G": 0, "B": 990, "M": 1, "cosq_idx": 7, "epsilon_idx": 2},
    "0.5% Noise": {"R": 5, "G": 0, "B": 995, "M": 1, "cosq_idx": 8, "epsilon_idx": 1},
    "0% Noise": {"R": 0, "G": 0, "B": 1, "M": 1, "cosq_idx": 0, "epsilon_idx": 0}
}

# COSQ options (bits/pixel)
COSQ_RATES = {
    "0.375 bpp": {"value": 24, "bpp": 0.375},
    "0.9 bpp": {"value": 58, "bpp": 0.9},
    "1.19 bpp": {"value": 76, "bpp": 1.19}
}

# COVQ options (block_size, rate) mapped to bits/pixel
COVQ_RATES = {
    "0.0625 bpp": {"block_size": 8, "rate": 4, "bpp": 0.0625},
    "0.125 bpp": {"block_size": 8, "rate": 8, "bpp": 0.125},
    "0.375 bpp": {"block_size": 4, "rate": 6, "bpp": 0.375},
    "0.5 bpp": {"block_size": 4, "rate": 8, "bpp": 0.5},
    "1 bpp": {"block_size": 2, "rate": 4, "bpp": 1},
    "2 bpp": {"block_size": 2, "rate": 8, "bpp": 2}
}

# Codebook paths
CODEBOOK_FILEPATH_COSQ = 'web_app/cosq_codebooks.json'
CODEBOOK_FILEPATH_LM = 'web_app/lloydmax_codebooks.json'
CODEBOOK_FILEPATH_COVQ = 'web_app/covq_codebooks.json'

def list_images_in_folder(folder_path="web_app/images"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def load_codebooks():
    codebooks = {}
    try:
        with open(CODEBOOK_FILEPATH_COSQ, 'r') as f:
            codebooks['cosq'] = json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Could not find {CODEBOOK_FILEPATH_COSQ}")
        codebooks['cosq'] = None

    try:
        with open(CODEBOOK_FILEPATH_LM, 'r') as f:
            codebooks['lm'] = json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Could not find {CODEBOOK_FILEPATH_LM}")
        codebooks['lm'] = None

    try:
        with open(CODEBOOK_FILEPATH_COVQ, 'r') as f:
            codebooks['covq'] = json.load(f)
    except FileNotFoundError:
        print(f"[WARNING] Could not find {CODEBOOK_FILEPATH_COVQ}")
        codebooks['covq'] = None

    return codebooks

CODEBOOKS = load_codebooks()

def process_all_methods(image_array, cosq_rate, covq_rate, noise_level, delta_value):
    """Process image using all four methods simultaneously"""
    h, w = image_array.shape
    
    # Get parameters
    params = NOISE_LEVELS[noise_level]
    delta = int(delta_value)
    
    # ===== COSQ Processing =====
    cosq_params = COSQ_RATES[cosq_rate]
    if CODEBOOKS['cosq']:
        comp_array_cosq, dcMean_c, dcStd_c, acStd_c = compress_image_with_codebook(
            image_array, CODEBOOKS['cosq'], cosq_params["value"], params["cosq_idx"], block_size=8
        )
        noisy_cosq = cosq_implement_channel(comp_array_cosq, params["R"], params["G"], params["B"], params["M"], delta)
        decoded_blocks_cosq = dequantize_image(
            noisy_cosq, CODEBOOKS['cosq'], cosq_params["value"], params["cosq_idx"], 8, dcMean_c, dcStd_c, acStd_c
        )
        cosq_reconstructed = create_montage(decoded_blocks_cosq, h, w, 8)
        psnr_cosq = cosq_compute_psnr(image_array, cosq_reconstructed)
    else:
        cosq_reconstructed = np.zeros_like(image_array)
        psnr_cosq = "N/A"
    
    # ===== Lloyd-Max Processing =====
    if CODEBOOKS['lm']:
        comp_array_lm, dcMean_lm, dcStd_lm, acStd_lm = compress_image_with_lloydmax_codebook(
            image_array, CODEBOOKS['lm'], cosq_params["value"], block_size=8
        )
        noisy_lm = cosq_implement_channel(comp_array_lm, params["R"], params["G"], params["B"], params["M"], delta)
        decoded_blocks_lm = dequantize_image_lloydmax(
            noisy_lm, CODEBOOKS['lm'], cosq_params["value"], 8, dcMean_lm, dcStd_lm, acStd_lm
        )
        lm_reconstructed = create_montage(decoded_blocks_lm, h, w, 8)
        psnr_lm = cosq_compute_psnr(image_array, lm_reconstructed)
    else:
        lm_reconstructed = np.zeros_like(image_array)
        psnr_lm = "N/A"
    
    # ===== COVQ Processing =====
    covq_params = COVQ_RATES[covq_rate]
    epsilons = [0, 0.005, 0.01, 0.05, 0.1]
    epsilon = epsilons[params["epsilon_idx"]]
    
    if CODEBOOKS['covq']:
        codebook_key = f"codebooks_{covq_params['block_size']}block_rate{covq_params['rate']}"
        vq_codebook_key = f"VQcodebooks_{covq_params['block_size']}block"
        
        loaded = {
            "mean": np.array(CODEBOOKS['covq'][codebook_key]["mean"]),
            "std": np.array(CODEBOOKS['covq'][codebook_key]["std"]),
            f"epsilon{params['epsilon_idx']}": np.array(CODEBOOKS['covq'][codebook_key][f"epsilon{params['epsilon_idx']}"])
        }
        
        loadedVQ = {
            "mean": np.array(CODEBOOKS['covq'][vq_codebook_key].get("mean", CODEBOOKS['covq'][codebook_key]["mean"])),
            "std": np.array(CODEBOOKS['covq'][vq_codebook_key]["std"]),
            f"rate{covq_params['rate']}": np.array(CODEBOOKS['covq'][vq_codebook_key][f"rate{covq_params['rate']}"])
        }
        
        grey_img_norm = (image_array - loaded["mean"]) / loaded["std"]
        num_centroids = pow(2, covq_params["rate"])
        init_b = list(range(num_centroids))
        
        # COVQ
        codebook = loaded[f"epsilon{params['epsilon_idx']}"]
        indices = encode_image_blocks(grey_img_norm, codebook, init_b, epsilon, delta, covq_params["block_size"])
        bitstream = indices_to_bitstream(indices, covq_params["rate"])
        channel_bitstream = covq_implement_channel(bitstream, params["R"], params["G"], params["B"], params["M"], delta)
        received_indices = bitstream_to_indices(channel_bitstream, covq_params["rate"])
        covq_reconstructed = decode_image_blocks(received_indices, codebook, covq_params["block_size"], image_array.shape)
        covq_reconstructed = covq_reconstructed * loaded["std"] + loaded["mean"]
        psnr_covq = covq_psnr(image_array, covq_reconstructed)
        
        # VQ
        codebookVQ = loadedVQ[f"rate{covq_params['rate']}"]
        indicesVQ = encode_image_blocks(grey_img_norm, codebookVQ, init_b, epsilon, delta, covq_params["block_size"])
        bitstreamVQ = indices_to_bitstream(indicesVQ, covq_params["rate"])
        channel_bitstreamVQ = covq_implement_channel(bitstreamVQ, params["R"], params["G"], params["B"], params["M"], delta)
        received_indicesVQ = bitstream_to_indices(channel_bitstreamVQ, covq_params["rate"])
        vq_reconstructed = decode_image_blocks(received_indicesVQ, codebookVQ, covq_params["block_size"], image_array.shape)
        vq_reconstructed = vq_reconstructed * loadedVQ["std"] + loadedVQ["mean"]
        psnr_vq = covq_psnr(image_array, vq_reconstructed)
    else:
        covq_reconstructed = np.zeros_like(image_array)
        vq_reconstructed = np.zeros_like(image_array)
        psnr_covq = "N/A"
        psnr_vq = "N/A"
    
    # Convert to PIL Images
    original_pil = Image.fromarray(image_array.astype(np.uint8), mode='L')
    cosq_pil = Image.fromarray(cosq_reconstructed.astype(np.uint8), mode='L') if isinstance(cosq_reconstructed, np.ndarray) else None
    lm_pil = Image.fromarray(lm_reconstructed.astype(np.uint8), mode='L') if isinstance(lm_reconstructed, np.ndarray) else None
    covq_pil = Image.fromarray(covq_reconstructed.astype(np.uint8), mode='L') if isinstance(covq_reconstructed, np.ndarray) else None
    vq_pil = Image.fromarray(vq_reconstructed.astype(np.uint8), mode='L') if isinstance(vq_reconstructed, np.ndarray) else None
    
    return (
        original_pil,
        cosq_pil, f"{psnr_cosq:.2f} dB" if isinstance(psnr_cosq, float) else psnr_cosq,
        lm_pil, f"{psnr_lm:.2f} dB" if isinstance(psnr_lm, float) else psnr_lm,
        covq_pil, f"{psnr_covq:.2f} dB" if isinstance(psnr_covq, float) else psnr_covq,
        vq_pil, f"{psnr_vq:.2f} dB" if isinstance(psnr_vq, float) else psnr_vq,
        f"COSQ {cosq_rate}",
        f"SQ {cosq_rate}",
        f"COVQ {covq_rate}",
        f"VQ {covq_rate}"
    )

def build_interface():
    custom_css = """
    .method-column {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        min-width: 300px !important;
    }
    .baseline-column {
        border: 2px solid #FF0000;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        min-width: 300px !important;
    }
    .results-row {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }
    .results-column {
        flex: 0 0 auto !important;
    }
    .method-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 10px;
    }
    @media (max-width: 768px) {
        .method-column, .baseline-column {
            min-width: 280px !important;
            max-width: 280px !important;
        }
        .image-container img {
            max-width: 100% !important;
            height: auto !important;
        }
    }
    """


    with gr.Blocks(title="Quantization Comparison", css=custom_css) as demo:
        gr.Markdown("# Image Compression Demo! ")
        gr.Markdown("Presenting a Channel-Optimized Scalar Quantizer (COSQ) and Channel-Optimized Vector Quantizer with their non-channel-optimized counterparts.")
        
        # ===== Top Section: Image Selection =====
        with gr.Row():
            with gr.Column():
                # Load default image
                try:
                    # Process default image
                    grey_img = cosq_grey_image_array("web_app/images/satelliteIMG01.jpg")
                    cropped_img = crop_to_square(grey_img)
                    default_pil = Image.fromarray(cropped_img.astype(np.uint8))
                    default_array = cropped_img
                except Exception as e:
                    print(f"Error loading default image: {e}")
                    # Fallback to black image
                    default_array = np.zeros((256, 256), dtype=np.uint8)
                    default_pil = Image.fromarray(default_array)
                
                # Image selection with live preview
                image_dropdown = gr.Dropdown(
                    label="Select Image",
                    choices=list_images_in_folder(),
                    value="satelliteIMG01.jpg" if "satelliteIMG01.jpg" in list_images_in_folder() else None
                )
                original_output = gr.Image(
                    label="Original (Grayscale)", 
                    type="pil",
                    height=300,
                    width=300,
                    elem_classes=["image-container"],
                    interactive=False,
                    value=default_pil  # Set default image
                )
                
                # Store the numpy array for processing
                image_array_state = gr.State(value=default_array)
                
                # Function to update image when selected
                def update_original_image(image_filename):
                    if image_filename:
                        image_path = os.path.join("images", image_filename)
                        grey_img = cosq_grey_image_array(image_path)
                        cropped_img = crop_to_square(grey_img)
                        pil_img = Image.fromarray(cropped_img.astype(np.uint8))
                        return pil_img, cropped_img
                    return default_pil, default_array
                
                # Connect dropdown to image display
                image_dropdown.change(
                    fn=update_original_image,
                    inputs=image_dropdown,
                    outputs=[original_output, image_array_state]
                )


        # ===== Middle Section: Parameters =====
        with gr.Row():
            # COSQ Parameters
            with gr.Column():
                gr.Markdown("### COSQ Parameters")
                cosq_rate_dropdown = gr.Dropdown(
                    label="Compression Rate (bits/pixel)",
                    choices=list(COSQ_RATES.keys()),
                    value="0.375 bpp"
                )
            
            # COVQ Parameters
            with gr.Column():
                gr.Markdown("### COVQ Parameters")
                covq_rate_dropdown = gr.Dropdown(
                    label="Compression Rate (bits/pixel)",
                    choices=list(COVQ_RATES.keys()),
                    value="0.375 bpp"
                )
        
        # Channel Parameters
        gr.Markdown("### Channel Parameters")
        with gr.Row():
            with gr.Column():
                noise_level_dropdown = gr.Dropdown(
                    label="Noise Level",
                    choices=list(NOISE_LEVELS.keys()),
                    value="10% Noise"
                )
            with gr.Column():
                delta_dropdown = gr.Dropdown(
                    label="Delta",
                    choices=["0", "5", "10", "50", "100", "1000"],
                    value="10"
                )
        
        # Run button
        with gr.Row():
            run_btn = gr.Button("Run Comparison", variant="primary")
        
        # ===== Results Section ==== #
        
        # Baseline Methods
        gr.Markdown("## Baseline Comparison")
        with gr.Row(elem_classes=["results-row"]):
            # SQ Results
            with gr.Column(elem_classes=["baseline-column", "results-column"], min_width=300):
                gr.Markdown("**SQ Baseline**", elem_classes=["method-title"])
                lm_output = gr.Image(
                    type="pil",
                    height=250,
                    width=250,
                    show_download_button=True,
                    elem_classes=["image-container"]
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        lm_label = gr.Textbox(label="Compression", value="SQ", interactive=False)
                    with gr.Column(scale=1):
                        lm_psnr = gr.Textbox(label="Quality (PSNR dB)", interactive=False)
            
            # VQ Results
            with gr.Column(elem_classes=["baseline-column", "results-column"], min_width=300):
                gr.Markdown("**VQ Baseline**", elem_classes=["method-title"])
                vq_output = gr.Image(
                    type="pil",
                    height=250,
                    width=250,
                    show_download_button=True,
                    elem_classes=["image-container"]
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vq_label = gr.Textbox(label="Compression", value="VQ", interactive=False)
                    with gr.Column(scale=1):
                        vq_psnr = gr.Textbox(label="Quality (PSNR dB)", interactive=False)
            
        # Main Methods
        gr.Markdown("## Channel Optimized Comparison")
        with gr.Row(elem_classes=["results-row"]):
            # COSQ Results
            with gr.Column(elem_classes=["method-column", "results-column"], min_width=300):
                gr.Markdown("**COSQ Results**", elem_classes=["method-title"])
                cosq_output = gr.Image(
                    type="pil",
                    height=250,
                    width=250,
                    show_download_button=True,
                    elem_classes=["image-container"]
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        cosq_label = gr.Textbox(label="Compression", value="COSQ 0.375 bpp", interactive=False)
                    with gr.Column(scale=1):
                        cosq_psnr = gr.Textbox(label="Quality (PSNR dB)", interactive=False)
            
            # COVQ Results
            with gr.Column(elem_classes=["method-column", "results-column"], min_width=300):
                gr.Markdown("**COVQ Results**", elem_classes=["method-title"])
                covq_output = gr.Image(
                    type="pil",
                    height=250,
                    width=250,
                    show_download_button=True,
                    elem_classes=["image-container"]
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        covq_label = gr.Textbox(label="Compression", value="COVQ 0.375 bpp", interactive=False)
                    with gr.Column(scale=1):
                        covq_psnr = gr.Textbox(label="Quality (PSNR dB)", interactive=False)

        run_btn.click(
            fn=process_all_methods,
            inputs=[
                image_array_state,  # Use the stored numpy array
                cosq_rate_dropdown,
                covq_rate_dropdown,
                noise_level_dropdown,
                delta_dropdown
            ],
            outputs=[
                original_output,  # This will be updated with the processed original
                cosq_output, cosq_psnr,
                lm_output, lm_psnr,
                covq_output, covq_psnr,
                vq_output, vq_psnr,
                cosq_label, lm_label,
                covq_label, vq_label
            ]
        )

    return demo

def main():
    demo = build_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()