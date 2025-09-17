from PIL import Image
import numpy as np
import os

# read image and convert rgb to ycbcr and yuv
def read_img(img_path: str):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_array = np.array(img).astype(np.float64) 
    
    # Extract R, G, B channels
    r = img_array[:, :, 0]
    g = img_array[:, :, 1] 
    b = img_array[:, :, 2]
    
    # Convert to YCbCr and YUV
    y, cb, cr, u, v = rgb_to_ycbcr_yuv(r, g, b)
    
    return r, g, b, y, cb, cr, u, v

# save each image as grayscale
def save_img(r, g, b, y, cb, cr, u, v, output_dir="output"):   
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dictionary of channels and their names
    channels = {
        'R': r.astype(np.uint8),
        'G': g.astype(np.uint8), 
        'B': b.astype(np.uint8),
        'Y': y,
        'Cb': cb,
        'Cr': cr,
        'U': u,
        'V': v
    }
    
    img_list = []
    
    # Save each channel as a grayscale image
    for channel_name, channel_data in channels.items():
        # Create PIL Image from numpy array
        img = Image.fromarray(channel_data, 'L')  # 'L' mode for grayscale
        
        # Save the image
        filename = f"{output_dir}/{channel_name}_channel.png"
        img.save(filename)
        img_list.append(filename)
        
        print(f"Saved {channel_name} channel to {filename}")
    
    return img_list

def rgb_to_ycbcr_yuv(r, g, b):
    # Calculate Y, Cb, Cr, U and V components
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

    u = -0.169 * r - 0.331 * g + 0.5 * b + 128
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128

    # Clip values to [0, 255] range and convert to 8-bit unsigned integers
    y = np.clip(y, 0, 255).astype(np.uint8)
    cb = np.clip(cb, 0, 255).astype(np.uint8)
    cr = np.clip(cr, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)

    return y, cb, cr, u, v

if __name__ == "__main__":
    # Read the image and get all channels
    r, g, b, y, cb, cr, u, v = read_img('./lena.png')
    
    # Save each channel as a grayscale image
    saved_files = save_img(r, g, b, y, cb, cr, u, v)