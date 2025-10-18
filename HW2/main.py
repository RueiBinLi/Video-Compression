import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

def read_img(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img

def gen_matrix_2D(gray_img):
    N, M = gray_img.shape
    matrix = np.zeros((N,N), dtype=float)

    C_u = np.ones(N) * np.sqrt(2 / N)
    C_u[0] = np.sqrt(1 / N)

    C_v = np.ones(M) * np.sqrt(2 / M)
    C_v[0] = np.sqrt(1 / M)

    print('start 2D-DCT')

    for u in range(N):
        for v in range(M):
            sum_val = 0.0
            for x in range(N):
                for y in range(M):
                    cos1 = np.cos((2 * x + 1) * np.pi * u / (2 * N))
                    cos2 = np.cos((2 * y + 1) * np.pi * v / (2 * M))

                    sum_val += gray_img[x, y] * cos1 * cos2
            
            matrix[u, v] = C_u[u] * C_v[v] * sum_val
    
    print('Finish 2D-DCT')
    return matrix

def DCT_2D(gray_img):
    start_time = time.time()
    T = gen_matrix_2D(gray_img)
    end_time = time.time()
    print(f'2D-DCT time: {end_time - start_time:.4f} seconds')

    T_log = np.log2(np.abs(T) + 1e-9)
    T_vis = cv2.normalize(T_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imwrite('2D_DCT_lena.png', T_vis)

    return T_vis

def gen_matrix_IDCT_2D(T_img):
    N, M = T_img.shape
    img_recon = np.zeros((N,N), dtype=float)

    C_u = np.ones(N) * np.sqrt(2 / N)
    C_u[0] = np.sqrt(1 / N)

    C_v = np.ones(N) * np.sqrt(2 / N)
    C_v[0] = np.sqrt(1 / N)

    for x in range(N):
        for y in range(M):
            sum_val = 0.0
            for u in range(N):
                for v in range(M):
                    cos1 = np.cos((2 * x + 1) * np.pi * u / (2 * N))
                    cos2 = np.cos((2 * y + 1) * np.pi * v / (2 * M))

                    sum_val += C_u[u] * C_v[v] * T_img[u, v] * cos1 * cos2
            img_recon[x, y] = sum_val

    return img_recon

def IDCT_2D(T_img):
    img_recon = gen_matrix_2D(T_img)
    img_recon = np.clip(img_recon, 0, 255).astype(np.uint8)

    cv2.imwrite('Reconstructed_lena.png', img_recon)
    return img_recon

def PSNR(ori_img, dct_img):
    mse = np.mean((ori_img - dct_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def gen_matrix_1D(N):
    matrix = np.zeros((N,N), dtype=float)

    C = np.ones(N) * np.sqrt(2 / N)
    C[0] = np.sqrt(1 / N)

    for u in range(N):
        for v in range(N):
            matrix[u, v] = C[u] * np.cos((2 * v + 1) * np.pi * u / (2 * N))
    return matrix

def DCT_1D(gray_img):
    rows, cols = gray_img.shape

    start_time = time.time()
    A = gen_matrix_1D(rows)
    B = gen_matrix_1D(cols).T
    G = gray_img.astype(np.float32)
    T = (A @ G) @ B
    end_time = time.time()
    print(f'Two 1D-DCT time: {end_time - start_time:.4f} seconds')

    T_log = np.log2(np.abs(T) + 1e-9)
    T_log = cv2.normalize(T_log, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('gray_lena.png', gray_img)
    cv2.imwrite('Two_2D_DCT_lena.png', T_log)

def main():
    gray_img = read_img('./lena.png')
    
    T_img = DCT_2D(gray_img)
    reconstruct_img = IDCT_2D(T_img)
    DCT_1D(gray_img)
    
    psnr = PSNR(ori_img=gray_img, dct_img=reconstruct_img)
    print(f'PSNR score: {psnr:.4f}')


if __name__ == '__main__':
    main()