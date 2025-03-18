def print_compressed_array(compressed_img_array):
    for i, block in enumerate(compressed_img_array):
        print(f"Block {i + 1}:")
        print(f"Lenth: {len(block)}")
        print(block)
        print()  # Add a blank line between blocks