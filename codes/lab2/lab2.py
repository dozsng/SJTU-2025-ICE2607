import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# ===============================
# 基于 goodFeaturesToTrack 的多尺度 Harris 角点检测
# ===============================
def multi_scale_harris(img, levels=6, scale_factor=0.75,
                       max_corners=9000, quality_level=0.00001,
                       min_distance=3, k=0.04, border_size=16):
    pyramid = [img]
    keypoints = []

    # ===== 构建图像金字塔 =====
    for _ in range(1, levels):
        h, w = pyramid[-1].shape[:2]
        new_img = cv2.resize(
            pyramid[-1],
            (int(w * scale_factor), int(h * scale_factor)),
            interpolation=cv2.INTER_LINEAR
        )
        pyramid.append(new_img)

    h0, w0 = img.shape[:2]  # 原图尺寸，用于边界判断

    # ===== 遍历每一层，提取角点 =====
    for level, im in enumerate(pyramid):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            useHarrisDetector=True,
            k=k
        )

        scale = 1 / (scale_factor ** level)  # 还原到原图比例

        if corners is not None:
            for pt in corners:
                x, y = pt.ravel()
                x_scaled, y_scaled = x * scale, y * scale

                # 边界过滤：保证计算 descriptor 时不会越界
                if (border_size <= x_scaled < w0 - border_size and
                    border_size <= y_scaled < h0 - border_size):
                    keypoints.append(cv2.KeyPoint(x_scaled, y_scaled, 3 * scale))

    return keypoints



# ===============================
# 梯度与方向图
# ===============================
def compute_gradients(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    angle = np.rad2deg(np.arctan2(Iy, Ix)) % 360
    return mag, angle


# ===============================
# 为关键点分配主方向
# ===============================
def assign_orientation(mag, angle, kp, radius=18, num_bins=36):
    x, y = kp.pt
    h, w = mag.shape
    histogram = np.zeros(num_bins, dtype=np.float32)
    bin_width = 360 / num_bins
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            xx, yy = int(x + j), int(y + i)
            if 0 <= xx < w and 0 <= yy < h:
                m = mag[yy, xx]
                a = angle[yy, xx]
                sigma = 0.5 * radius
                weight = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                bin_idx = int(a // bin_width) % num_bins
                histogram[bin_idx] += m * weight
    main_angle = np.argmax(histogram) * bin_width
    return main_angle


# ===============================
# 描述子计算（旋转对齐 + 双线性插值 + 高斯加权 + 阈值截断 + 双归一化）
# ===============================
def compute_sift_descriptor(mag, angle, kp, patch_size=16, num_bins=8, clip_val=0.2):
    x, y = kp.pt
    main_angle = assign_orientation(mag, angle, kp)
    cos_t, sin_t = np.cos(np.radians(main_angle)), np.sin(np.radians(main_angle))
    half = patch_size // 2
    desc = np.zeros((4, 4, num_bins), dtype=np.float32)
    bin_width = 360.0 / num_bins
    sigma = 0.5 * patch_size

    for i in range(-half, half):
        for j in range(-half, half):
            # 旋转对齐（注意 i->y, j->x）
            rot_x = j * cos_t - i * sin_t
            rot_y = j * sin_t + i * cos_t
            xx_f = x + rot_x
            yy_f = y + rot_y

            # 边界检查（注意使用 floor/ceil 时保持安全）
            if not (0 <= xx_f < mag.shape[1] and 0 <= yy_f < mag.shape[0]):
                continue

            # 推荐：对 mag 做双线性插值会更好。这里先用 nearest
            xx, yy = int(xx_f), int(yy_f)
            m = mag[yy, xx]
            a = (angle[yy, xx] - main_angle) % 360

            # 高斯权重
            weight = np.exp(-(rot_x**2 + rot_y**2) / (2 * sigma**2))
            m_weighted = m * weight

            # 浮点小块坐标（修正：不要先 int）
            cell_size = patch_size / 4.0
            bx_f = (rot_x + half) / cell_size
            by_f = (rot_y + half) / cell_size

            bx0 = int(np.floor(bx_f))
            by0 = int(np.floor(by_f))
            dx = bx_f - bx0
            dy = by_f - by0

            # 空间权重四个（双线性）
            w = [(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy]
            bx_idx = [bx0, bx0+1, bx0, bx0+1]
            by_idx = [by0, by0, by0+1, by0+1]

            # 方向插值（线性，带环绕）
            bin_f = a / bin_width
            b0 = int(np.floor(bin_f)) % num_bins
            b1 = (b0 + 1) % num_bins
            wb1 = bin_f - np.floor(bin_f)
            wb0 = 1.0 - wb1
            bin_idx = [b0, b1]
            w_dir = [wb0, wb1]

            # 累加
            for ii in range(4):
                bx_i, by_i = bx_idx[ii], by_idx[ii]
                if 0 <= bx_i < 4 and 0 <= by_i < 4:
                    for bb, wb in zip(bin_idx, w_dir):
                        desc[by_i, bx_i, bb] += m_weighted * w[ii] * wb

    # 展平 + 截断 + 双归一化
    desc = desc.flatten()
    norm = np.linalg.norm(desc) + 1e-7
    desc = desc / norm
    desc = np.clip(desc, 0, clip_val)
    norm2 = np.linalg.norm(desc) + 1e-7
    desc = desc / norm2
    return desc



# ===============================
# 特征匹配（BFMatcher）
# ===============================
def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append((m.trainIdx, m.queryIdx))
    return good


# ===============================
# 自实现 Harris + SIFT 描述子匹配
# ===============================
def custom_sift_match(target_path, dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    target = cv2.imread(target_path)
    gray_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    kps_t = multi_scale_harris(target)
    mag_t, ang_t = compute_gradients(gray_t)
    descs_t = np.array([compute_sift_descriptor(mag_t, ang_t, kp, patch_size=int(4 * kp.size)) for kp in kps_t])

    best_img, best_score, best_kps, best_matches = None, 0, None, None

    for name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = multi_scale_harris(img)
        mag, ang = compute_gradients(gray)
        descs = np.array([compute_sift_descriptor(mag, ang, kp, patch_size=int(4 * kp.size)) for kp in kps])

        matches = match_features(descs_t, descs)
        print(f"{name}: 匹配数量 {len(matches)}")

        if len(matches) > best_score:
            best_img, best_score, best_kps, best_matches = img, len(matches), kps, matches

    if best_img is not None:
        matched_img = cv2.drawMatches(
            best_img, best_kps,
            target, kps_t,
            [cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0) for i, j in best_matches],
            None, flags=2
        )
        # 统一使用 Matplotlib 保存为 RGB
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
        plt.title("Custom Harris + SIFT")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "custom_harris_sift.png"))
        plt.close()
        print("匹配结果已保存！")


# ===============================
# OpenCV SIFT匹配（保持原样）
# ===============================
def opencv_sift_match(target_path, dataset_dir, output_dir):
    target = cv2.imread(target_path)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp_t, desc_t = sift.detectAndCompute(gray_target, None)

    best_img, best_score, best_matches, best_kp = None, 0, None, None
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    for name in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        matches = bf.knnMatch(desc_t, desc, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        print(f"{name}: OpenCV匹配数量 {len(good)}")
        if len(good) > best_score:
            best_img, best_score, best_matches, best_kp = img, len(good), good, kp

    if best_img is not None:
        reversed_matches = [
            cv2.DMatch(_queryIdx=m.trainIdx, _trainIdx=m.queryIdx, _imgIdx=m.imgIdx, _distance=m.distance) for m in
            best_matches]
        result = cv2.drawMatches(best_img, best_kp, target, kp_t, reversed_matches, None, flags=2)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("OpenCV SIFT")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "opencv_sift_result.png"))
        plt.close()
        print("OpenCV SIFT匹配结果已保存")


# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    cwd = os.path.dirname(__file__)
    dataset_dir = os.path.join(cwd, "dataset")
    target_path = os.path.join(cwd, "target.jpg")
    output_dir = os.path.join(cwd, "output")
    os.makedirs(output_dir, exist_ok=True)
    print("开始自实现SIFT匹配...")
    custom_sift_match(target_path, dataset_dir, output_dir)
    print("开始OpenCV SIFT匹配...")
    opencv_sift_match(target_path, dataset_dir, output_dir)
    print("所有匹配结果已保存至 output 文件夹")
