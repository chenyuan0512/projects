import cv2
import numpy as np
import tkinter as tk  # 用來取得螢幕解析度
import matplotlib.pyplot as plt
from camera_connect_and_find_reflected_center.M6_take_photo_and_connect import check_connection, capture_image
import os
import glob
import time


class FindReflectedCenter():
    def __init__(self):
        self.pic_folder_path = 'C:\\Users\\Sulab\\Desktop\\Program_ChenYuan_Version\\photo_from_M6\\'  # 看照片的路徑
        self.patch_size = 4
        self.window = self.Hann_2d(self.patch_size, self.patch_size)
        self.image = None
    

    def Hann_2d(self, h, w):
        # 因為照片切成正方形，所以patch_size的長寬是固定的
        wy = np.hanning(h).astype(np.float64)
        wx = np.hanning(w).astype(np.float64)
        window = np.outer(wy, wx).astype(np.float64)
        return window


    def take_picture(self):
        check_connection()
        # self.camera_function["check_connection"]
        now_str = time.strftime("%Y%m%d_%H%M%S")  # 取得現在的時分
        file_path = "c//Users//Sulab//Desktop//Program_ChenYuan_Version//photo_from_M6"
        capture_image(file_path, f"{now_str}.jpg", "Mark")
        time.sleep(2)

    
    # 讀取最後一張照片
    def read_image(self):
        # 照片的路徑
        # 取得所有 .jpg 檔案（使用絕對路徑）
        jpg_files = glob.glob(os.path.join(self.pic_folder_path, '*.jpg'))

        # 依檔名排序（只看檔名本身不看路徑）
        jpg_files_sorted = sorted(jpg_files, key=lambda x: os.path.basename(x))

        image = cv2.imread(jpg_files_sorted[-1])  # 讀取最後一張圖片

        if image is None:
            raise IOError(f"讀圖失敗：{self.pic_folder_path}")

        self.image = image
        return image


    # 照片調整成每小張都一樣大
    def reshape_image(self):
        h, w = self.image.shape[:2]
        copy_h = h - h % self.patch_size
        copy_w = w - w % self.patch_size
        image_cropped = self.image[:copy_h, :copy_w]
        return image_cropped
    

    def process_image(self, image_cropped):

        image = cv2.medianBlur(image_cropped, 5)

        # === 只取出紅色遮罩 ===
        b, g, r = cv2.split(image) # 將照片分成R、G、B三個通道

        # 建立紅色遮罩：紅高、綠藍低（排除白光或雜訊）
        red_mask = cv2.inRange(r, 60, 255)  # 取出照片的r值，若>=50，就設成255，<50就設成0
        non_red = cv2.inRange(cv2.max(g, b), 0, 100)  # 取出g和b中較大的值，如果在0到100之間，回傳255，否則回傳0
        thresh = cv2.bitwise_and(red_mask, non_red)  # 只有兩個都是255才會是255，否則為0
        
        return thresh


    # 照片切成很多小塊
    def cut_image_into_patches_and_fft(self,image_thresh):
        h, w = image_thresh.shape[:2]
        nH, nW = h // self.patch_size, w // self.patch_size
        # Hc, Wc = nH * patch_size, nW * patch_size

        # 裁齊可整除邊長
        # img = image[:Hc, :Wc].astype(np.float64, copy=False)

        # 以 reshape+swapaxes 方式得到所有 patch： (nH, nW, p, p) → (N, p, p)
        patches = image_thresh.reshape(nH, self.patch_size, nW, self.patch_size).swapaxes(1, 2).reshape(-1, self.patch_size, self.patch_size)

        # 去均值（每塊各自減掉均值）
        patches = patches - patches.mean(axis=(1, 2), keepdims=True)

        # 乘窗（自動廣播到 (N, p, p)）
        patches = patches * self.window[None, :, :]

        # 2D FFT（全向量化）：(N, p, p)
        F = np.fft.fft2(patches, axes=(-2, -1))
        P = (F.real * F.real + F.imag * F.imag)

        # 去除 DC（第 0 列/行）
        P[:, 0, :] = 0
        P[:, :, 0] = 0

        # 去除 Nyquist 行/列（偶數長度才有 Nyquist 索引 = p//2）
        if self.patch_size % 2 == 0:
            ny = self.patch_size // 2
            
            P[:, ny, :] = 0
            P[:, :, ny] = 0

        # 建立每塊中心的 (x, y) 影像座標（以像素為單位）
        ys_grid, xs_grid = np.indices((nH, nW))
        ys = ((ys_grid + 0.5) * self.patch_size).reshape(-1).astype(np.float64)
        xs = ((xs_grid + 0.5) * self.patch_size).reshape(-1).astype(np.float64)

        return P, xs, ys

    

    # 算權重
    def calculate_inertia(self, fft_patches_array, xs, ys, epsilon = 1e-9, confidence_threshold_value = 0.5):
        
        P = fft_patches_array  # (N, p, p)
        N, p, _ = P.shape

        u = np.fft.fftfreq(p)  # [-0.5, ..., 0, ..., +0.5)
        v = np.fft.fftfreq(p)
        
        V, U = np.meshgrid(v, u, indexing="ij")  # (p, p)
        U = U[None, :, :]
        V = V[None, :, :]

        # 加權總能量與一階/二階矩（全部向量化）
        S   = np.sum(P, axis=(-2, -1), dtype=np.float64)                    # (N,)
        Sx  = np.sum(P * U, axis=(-2, -1), dtype=np.float64)                # (N,)
        Sy  = np.sum(P * V, axis=(-2, -1), dtype=np.float64)                # (N,)
        Sxx = np.sum(P * (U * U), axis=(-2, -1), dtype=np.float64)          # (N,)
        Syy = np.sum(P * (V * V), axis=(-2, -1), dtype=np.float64)          # (N,)
        Sxy = np.sum(P * (U * V), axis=(-2, -1), dtype=np.float64)          # (N,)

        # 為避免除 0，用 S_safe（也比你原式 Sxx - Sx^2/S 更穩定）
        S_safe = S + epsilon
        mu_x = Sx / S_safe
        mu_y = Sy / S_safe

        # 中心二階矩（相當於加權 covariance 的對角與互相關；不再乘以 S）歸一化
        Mx2 = Sxx / S_safe - mu_x * mu_x
        My2 = Syy / S_safe - mu_y * mu_y
        Mxy = Sxy / S_safe - mu_x * mu_y

        # 主軸角度（標準公式）：theta = 0.5 * atan2(2Mxy, My2 - Mx2)
        theta = 0.5 * np.arctan2(2.0 * Mxy, (My2 - Mx2))  # 減 pi/2 是因為要垂直於條紋方向
        denom = (Mx2 + My2)
        numer = np.sqrt((My2 - Mx2) * (My2 - Mx2) + 4.0 * Mxy * Mxy)
        C = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > epsilon) # 論文公式 (7)

        valid_mask = (S > epsilon) & np.isfinite(theta) & np.isfinite(C) & (C >= confidence_threshold_value)

        # 將所有需要下游使用的量一併過濾，確保 shape 一致
        theta_k = theta[valid_mask]
        Ck      = C[valid_mask]
        xs_f    = xs[valid_mask]
        ys_f    = ys[valid_mask]

        return Ck, theta_k, xs_f, ys_f
    

    # 論文公式10~14
    def find_center_point_parameters(self, Ck, theta_k, xs, ys, threshold_value = 1e-12):
        S1 = np.sum(Ck * np.sin(theta_k) ** 2, dtype=np.float64)
        S2 = np.sum(Ck * np.sin(theta_k) * np.cos(theta_k), dtype=np.float64)
        S3 = np.sum(Ck * (xs * np.sin(theta_k) ** 2 - ys * np.cos(theta_k) * np.sin(theta_k)), dtype=np.float64)
        S4 = np.sum(Ck * np.cos(theta_k) ** 2, dtype=np.float64)
        S5 = np.sum(Ck * (xs * np.cos(theta_k) * np.sin(theta_k) - ys * np.cos(theta_k) ** 2), dtype=np.float64)

        den = S2 ** 2 - S1 * S4
        den_safe = np.where(np.abs(den) < threshold_value, np.nan, den) # 如果小於threshold就設成nan，避免除0

        Xc = (S2 * S5 - S3 * S4) / den_safe
        Yc = (S1 * S5 - S2 * S3) / den_safe

        # 防 NaN
        if not np.isfinite(Xc) or not np.isfinite(Yc):
            return self.image.copy(), (np.nan, np.nan)

        image_final = cv2.circle(self.image, (int(Xc), int(Yc)), 5, (255, 255, 255), 10)
        center = (int(Xc), int(Yc))
        print(center)

        return image_final, center
    
    # 顯示照片
    def show_image(self, image_final, save_pic:bool):
        # 自動偵測螢幕尺寸
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # 自動縮放圖片
        h, w = image_final.shape[:2]
        scale_w = screen_width / w
        scale_h = screen_height / h
        scale = min(scale_w * 0.9, scale_h * 0.9)  # 再打 9 折，避免佔滿整個螢幕
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_image = cv2.resize(image_final, (new_w, new_h), interpolation=cv2.INTER_AREA)
        

        # 顯示圖片
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", new_w, new_h)
        cv2.imwrite('ver4.jpg', resized_image) if save_pic else None
        cv2.imshow("image", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
