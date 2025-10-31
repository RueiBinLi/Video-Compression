import numpy as np
import cv2
import time
import math

# --- 1. Helper Functions ---

def calculate_sad(block1, block2):
    """
    Calculates the Sum of Absolute Differences (SAD) between two blocks.
    Blocks are expected to be NumPy arrays.
    """
    return np.sum(np.abs(np.subtract(block1, block2, dtype=np.float32)))

def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    # Ensure images are float type for calculation
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        # MSE is zero means no noise, PSNR is infinite
        return float('inf')
        
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

# --- 2. Motion Estimation Algorithms ---

def full_search(ref_frame, cur_frame, block_size, search_range):
    """
    Performs Full Search block matching motion estimation.
    """
    # Pad the reference frame to handle blocks near the border
    pad = search_range
    ref_padded = cv2.copyMakeBorder(ref_frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    height, width = cur_frame.shape
    predicted_frame = np.zeros((height, width), dtype=np.uint8)
    
    # Iterate over the current frame in blocks
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Get the current block
            cur_block = cur_frame[y:y+block_size, x:x+block_size]
            
            min_sad = float('inf')
            best_mv = (0, 0) # (dx, dy)
            
            # --- Full Search Loop ---
            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    
                    # Get the corresponding block from the padded reference frame
                    ref_y = y + pad + dy
                    ref_x = x + pad + dx
                    ref_block = ref_padded[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                    
                    sad = calculate_sad(cur_block, ref_block)
                    
                    if sad < min_sad:
                        min_sad = sad
                        best_mv = (dx, dy)
            
            # --- Motion Compensation ---
            # Once best MV is found, copy the block from ref_frame
            ref_y_comp = y + best_mv[1]
            ref_x_comp = x + best_mv[0]
            
            # Clamp coordinates to stay within original ref_frame boundaries
            ref_y_comp = max(0, min(ref_y_comp, height - block_size))
            ref_x_comp = max(0, min(ref_x_comp, width - block_size))
            
            predicted_block = ref_frame[ref_y_comp:ref_y_comp+block_size, ref_x_comp:ref_x_comp+block_size]
            predicted_frame[y:y+block_size, x:x+block_size] = predicted_block
            
    return predicted_frame

def three_step_search(ref_frame, cur_frame, block_size, search_range):
    """
    Performs Three-Step Search (TSS) block matching motion estimation.
    """
    pad = search_range
    ref_padded = cv2.copyMakeBorder(ref_frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    height, width = cur_frame.shape
    predicted_frame = np.zeros((height, width), dtype=np.uint8)
    
    # Define the 9 search points (relative to center)
    search_points = [(0, 0), 
                     (0, 1), (0, -1), (1, 0), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            cur_block = cur_frame[y:y+block_size, x:x+block_size]
            
            # Start search from center (0,0 MV)
            center_mv = (0, 0) # (dx, dy)
            step = search_range // 2 # Initial step size (e.g., 4 for [+-8])
            
            min_sad = float('inf')
            
            # --- Three-Step Search Loop ---
            while step >= 1:
                best_mv_step = center_mv
                
                for dp in search_points:
                    # Calculate the MV to test: center_mv + (search_point * step)
                    test_dx = center_mv[0] + dp[0] * step
                    test_dy = center_mv[1] + dp[1] * step

                    # Skip if this MV is outside the overall search range
                    if abs(test_dx) > search_range or abs(test_dy) > search_range:
                        continue
                        
                    # Get the ref block from the padded frame
                    ref_y = y + pad + test_dy
                    ref_x = x + pad + test_dx
                    ref_block = ref_padded[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                    
                    sad = calculate_sad(cur_block, ref_block)

                    if sad < min_sad:
                        min_sad = sad
                        best_mv_step = (test_dx, test_dy)
                
                # Update the center for the next, smaller step
                center_mv = best_mv_step
                step //= 2
            
            best_mv = center_mv
            
            # --- Motion Compensation ---
            ref_y_comp = y + best_mv[1]
            ref_x_comp = x + best_mv[0]
            
            ref_y_comp = max(0, min(ref_y_comp, height - block_size))
            ref_x_comp = max(0, min(ref_x_comp, width - block_size))
            
            predicted_block = ref_frame[ref_y_comp:ref_y_comp+block_size, ref_x_comp:ref_x_comp+block_size]
            predicted_frame[y:y+block_size, x:x+block_size] = predicted_block
            
    return predicted_frame

# --- 3. Main Execution & Experiments ---

if __name__ == "__main__":
    # --- Setup ---
    try:
        ref_frame = cv2.imread('one_gray.png', cv2.IMREAD_GRAYSCALE)
        cur_frame = cv2.imread('two_gray.png', cv2.IMREAD_GRAYSCALE)
        
        if ref_frame is None or cur_frame is None:
            raise FileNotFoundError
            
    except FileNotFoundError:
        print("Error: Could not read 'one_gray.png' or 'two_gray.png'.")
        print("Please make sure the images are in the same directory as the script.")
        exit()

    BLOCK_SIZE = 8
    
    print("--- Experiment 1: Full Search (FS) with Varying Search Range ---")
    search_ranges = [8, 16, 32]
    fs_results = {}

    for r in search_ranges:
        print(f"\nRunning FS with search range [+-{r}]...")
        start_time = time.time()
        
        # --- ME and MC ---
        pred_frame_fs = full_search(ref_frame, cur_frame, BLOCK_SIZE, r)
        
        end_time = time.time()
        runtime_fs = end_time - start_time
        
        # --- Analysis ---
        psnr_fs = calculate_psnr(cur_frame, pred_frame_fs)
        residual_fs = cv2.absdiff(cur_frame, pred_frame_fs) # abs(cur - pred)
        
        # Save outputs
        cv2.imwrite(f'reconstructed_fs_r{r}.png', pred_frame_fs)
        cv2.imwrite(f'residual_fs_r{r}.png', residual_fs)
        
        print(f"FS [+-{r}] Results:")
        print(f"  PSNR: {psnr_fs:.4f} dB")
        print(f"  Runtime: {runtime_fs:.4f} seconds")
        
        fs_results[r] = (psnr_fs, runtime_fs)

    
    print("\n--- Experiment 2: Full Search (FS) vs. Three-Step Search (TSS) ---")
    # We will use the [+-8] range for a fair comparison
    r_compare = 8 
    
    print(f"\nRunning TSS with search range [+-{r_compare}]...")
    start_time_tss = time.time()
    
    # --- ME and MC ---
    pred_frame_tss = three_step_search(ref_frame, cur_frame, BLOCK_SIZE, r_compare)
    
    end_time_tss = time.time()
    runtime_tss = end_time_tss - start_time_tss
    
    # --- Analysis ---
    psnr_tss = calculate_psnr(cur_frame, pred_frame_tss)
    residual_tss = cv2.absdiff(cur_frame, pred_frame_tss)
    
    cv2.imwrite(f'reconstructed_tss_r{r_compare}.png', pred_frame_tss)
    cv2.imwrite(f'residual_tss_r{r_compare}.png', residual_tss)
    
    print("--- Comparison Summary ---")
    print(f"Algorithm (Range [+-{r_compare}]): \tPSNR (dB) \tRuntime (s)")
    print(f"Full Search: \t\t\t{fs_results[r_compare][0]:.4f} \t{fs_results[r_compare][1]:.4f}")
    print(f"Three-Step Search: \t\t{psnr_tss:.4f} \t{runtime_tss:.4f}")