# Lab2-SIFT尺度不变特征变换

## 1. 实验概览

### 1.1 实验目的

本实验旨在理解并掌握 SIFT（尺度不变特征变换）算法的原理与实现方法，通过自编程实现多尺度 Harris 角点检测结合自定义 SIFT 描述子的方法，并与 OpenCV 自带 SIFT 算法进行匹配效果对比，从而加深对特征提取、描述及匹配机制的理解。

### 1.2 实验原理

1. SIFT 算法原理
   SIFT（Scale-Invariant Feature Transform）由 David Lowe 提出，是一种在尺度空间中提取稳定关键点并计算其描述子的算法。SIFT 特征具有旋转、尺度及亮度变化不变性。其核心步骤包括：
   (1) 检测尺度空间极值点（通常通过高斯差分 DoG 实现）；
   (2) 精确定位关键点并剔除低对比度点；
   (3) 为关键点分配主方向以获得旋转不变性；
   (4) 计算关键点邻域内的梯度方向直方图以形成描述子。
2. Harris 角点检测
   Harris 角点检测通过计算图像局部灰度变化矩阵的特征值，检测强度变化显著的像素点作为角点。在本实验中，为实现多尺度鲁棒性，使用图像金字塔在多个缩放层次上提取角点。

## 2. 算法设计与实现

整体流程如下：

1. `multi_scale_harris`: 基于`goodFeaturesToTrack`的多尺度 Harris 检测，构建高斯金字塔，使用多尺度 Harris 算法提取关键点。
   
   `·` 在构建高斯金字塔时，首先确定金字塔的层数`levels`和缩放比例`scale_factor`，然后使用`cv2.resize()`函数进行图像的缩放（这里仅对图像做了下缩放），然后将图像存入金字塔中，得到高斯金字塔。其中`cv2.resize`函数中插值方法使用了`cv2.INTER_LANCZOS4`的Lanczos插值方法以做到高质量的图像缩放：
   
   ```python
   def multi_scale_harris(img, levels=8, scale_factor=0.75,
                          max_corners=10000, quality_level=0.00001, min_distance=2, k=0.04):
       pyramid = [img]
       keypoints = []
   
       # 构建图像金字塔
       for _ in range(1, levels):
           h, w = pyramid[-1].shape[:2]
           new_img = cv2.resize(pyramid[-1], (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_LINEAR)
           pyramid.append(new_img)
   ```
   
   `·` 对高斯金字塔中每一层的图像转换成灰色图像，然后使用`cv2.goodFeaturesToTrack()`获取Harris角点，得到每一层的角点坐标。
   
   ```python
       # 遍历每一层
       for level, im in enumerate(pyramid):
           gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
           gray = np.float32(gray)
   
           # 使用 goodFeaturesToTrack 获取 Harris 角点
           corners = cv2.goodFeaturesToTrack(
               gray,
               maxCorners=max_corners,
               qualityLevel=quality_level,
               minDistance=min_distance,
               useHarrisDetector=True,
               k=k
           )
   ```
   
   `·`将在不同金字塔层上检测到的 Harris 角点坐标映射回原图坐标系，并生成对应的 `cv2.KeyPoint` 对象，保存到总关键点列表中。
   
   ```python
           scale = 1 / (scale_factor ** level)  # 金字塔缩放回原图大小
   
           if corners is not None:
               for pt in corners:
                   x, y = pt.ravel()
                   keypoints.append(cv2.KeyPoint(x * scale, y * scale, 3 * scale))
   ```

2. `compute_gradients`计算图像梯度与方向。其中高斯模糊`cv2.GaussianBlur` 是前置预处理，减少噪声造成的“假角点”；使用Sobel算子，对关键点周围的16x16的区域进行梯度计算，得到梯度幅度`mag`和方向`angle`，并使用`np.rad2deg()`将方向转换到0-360度之间
   
   ```python
   def compute_gradients(gray):
       gray = cv2.GaussianBlur(gray, (3, 3), 0.5)
       Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
       Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
       mag = np.sqrt(Ix ** 2 + Iy ** 2)
       angle = np.rad2deg(np.arctan2(Iy, Ix)) % 360
       return mag, angle
   ```

3. `assign_orientation`在关键点 kp 的邻域内统计梯度方向直方图，找出该关键点的主方向（主方向用于把描述子旋转到物体坐标系，从而获得旋转不变性）。
   
   - 其中`weight`为**高斯空间权重**，靠近关键点的像素权重更大。这里 `sigma = 0.5 * radius`，所以权重是 `exp(-(r^2)/(2*sigma^2))`。选择 `sigma` 的目的是限制贡献主要来自关键点邻域中心。
   
   - `np.argmax(histogram)` 返回直方图最大值索引，`* bin_width` 转换为角度。
   
   - 返回 **关键点的主方向**。
   
   ```python
   def assign_orientation(mag, angle, kp, radius=18, num_bins=36):
       x, y = kp.pt
   
       sigma = 0.5 * radius
       weight = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
   
       main_angle = np.argmax(histogram) * bin_width
       return main_angle
   ```

4. `compute_sift_descriptor`计算 SIFT 描述子（旋转对齐 + 双线性插值 + 高斯加权 + 阈值截断 + 双归一化）。
   
   ```python
   def compute_sift_descriptor(mag, angle, kp, patch_size=16, num_bins=8, clip_val=0.25):
   
     main_angle = assign_orientation(mag, angle, kp)
     cos_t, sin_t = np.cos(np.radians(main_angle)), np.sin(np.radians(main_angle))
   
     for i in range(-half, half):
         for j in range(-half, half):
             # 旋转对齐
             rot_x = j * cos_t - i * sin_t
             rot_y = j * sin_t + i * cos_t
             xx, yy = int(x + rot_x), int(y + rot_y)
   
             if 0 <= xx < mag.shape[1] and 0 <= yy < mag.shape[0]:
                 m = mag[yy, xx]
                 a = (angle[yy, xx] - main_angle) % 360
   
                 # 高斯加权
                 weight = np.exp(-(rot_x**2 + rot_y**2) / (2 * sigma**2))
                 m_weighted = m * weight
   ```
   
   把当前像素分配到 4×4 的空间网格，每个像素不只影响一个 cell，而是根据距离分布到相邻 4 个 cell（双线性插值）。
   
   ```python
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
    
   ```
   
   对每个像素：
   
   根据空间插值权重 w[ii]，方向插值权重 wb，高斯权重 weight，梯度幅值 m；
   → 综合累加到对应的直方图 bin。
   
   结果：每个 cell（4×4）都有一个方向直方图（8 维）。
   
   ```python
   for ii in range(4):
       bx_i, by_i = bx_idx[ii], by_idx[ii]
       if 0 <= bx_i < 4 and 0 <= by_i < 4:
       for bb, wb in zip(bin_idx, w_dir):
       desc[by_i, bx_i, bb] += m_weighted * w[ii] * wb
   ```
   
   
   

5. match_features`对两张图像的 SIFT 描述子集合` desc1`和`desc2` 进行匹配，  
   返回通过 **Lowe 比率检测 (Lowe’s ratio test)** 筛选后的匹配点索引对。
   
   - `BFMatcher` = **Brute Force Matcher（暴力匹配器），工作机制：对 `desc1` 中的每个特征，依次计算它与 `desc2` 中所有特征的距离，使用欧氏距离 (`NORM_L2`) 比较两个描述子的相似程度，由于 SIFT 描述子已做归一化，因此欧氏距离可以直接反映特征相似度。
   
   ```python
   def match_features(desc1, desc2, ratio=0.75):
       bf = cv2.BFMatcher(cv2.NORM_L2)
       matches = bf.knnMatch(desc1.astype(np.float32), desc2.astype(np.float32), k=2)
   ```
   
   - David Lowe 在原始 SIFT 论文中提出：
     
     > “如果第一匹配的距离明显小于第二匹配的距离，则该匹配可信。”
     
     换句话说：  
     若一个特征在另一图像中找到了两个候选匹配点，  
     只有当最优匹配 **明显优于** 第二优匹配时，  
     才认为它是“稳定匹配”。
   
   ```python
   for m, n in matches:
       if m.distance < ratio * n.distance:
           good.append((m.trainIdx, m.queryIdx))
   ```

6. `custom_sift_match` 

7. 
