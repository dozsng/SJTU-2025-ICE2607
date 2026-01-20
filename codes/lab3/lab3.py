import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from time import time

def compute_color_histogram(image):
    image = cv2.resize(image, (200, 200))
    h, w = image.shape[:2]
    mid_h = h // 2
    mid_w = w // 2
    regions = [image[0:mid_h, 0:mid_w], image[0:mid_h, mid_w:w],
               image[mid_h:h, 0:mid_w], image[mid_h:h, mid_w:w]]
    hists = []
    for region in regions:
        b, g, r = cv2.split(region)
        b_energy = np.sum(b)
        g_energy = np.sum(g)
        r_energy = np.sum(r)
        total_energy = b_energy + g_energy + r_energy
        hist = [b_energy / total_energy, g_energy / total_energy, r_energy / total_energy]
        bins = np.percentile(hist, [33, 66])
        quantized = np.digitize(hist, bins)
        hists.extend(list(quantized[:3]))
    return np.array(hists[:12], dtype=int)

def lsh_hash(p, C, index_set):
    g = []
    for t in index_set:
        i = (t - 1) // C
        v_t = 1 if t <= i * C + p[i] else 0
        g.append(v_t)
    return tuple(g)

def build_lsh_table(data, C=3, L=4, m=6, seed=0):
    np.random.seed(seed)
    d, d_prime = data.shape[1], data.shape[1]*C
    hash_tables, index_sets = [], []
    for _ in range(L):
        idx = np.random.choice(np.arange(1, d_prime+1), size=m, replace=False)
        index_sets.append(sorted(idx))
        table = defaultdict(list)
        for idx_data, p in enumerate(data):
            key = lsh_hash(p, C, idx)
            table[key].append(idx_data)
        hash_tables.append(table)
    return hash_tables, index_sets

def query_lsh(query, hash_tables, index_sets, C):
    candidates = set()
    for table, idx_set in zip(hash_tables, index_sets):
        key = lsh_hash(query, C, idx_set)
        candidates.update(table.get(key, []))
    return list(candidates)

def l1_distance(a, b):
    return np.sum(np.abs(a - b))

def brute_force_search(query, data):
    dists = [l1_distance(query, x) for x in data]
    return np.argmin(dists)

def lsh_search(query, data, hash_tables, index_sets, C):
    candidates = query_lsh(query, hash_tables, index_sets, C)
    if not candidates:
        return None
    dists = [l1_distance(query, data[i]) for i in candidates]
    return candidates[np.argmin(dists)]

def visualize_results(query_img_path, nn_img_path, lsh_img_path=None, save_path=None):
    # 读取图像
    query_img = cv2.imread(query_img_path)
    nn_img = cv2.imread(nn_img_path)
    lsh_img = cv2.imread(lsh_img_path) if lsh_img_path else None

    # 转 RGB
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    nn_img = cv2.cvtColor(nn_img, cv2.COLOR_BGR2RGB)
    if lsh_img is not None:
        lsh_img = cv2.cvtColor(lsh_img, cv2.COLOR_BGR2RGB)

    # 画图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(query_img)
    plt.title('Target Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(nn_img)
    plt.title('Brute Force NN')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    if lsh_img is not None:
        plt.imshow(lsh_img)
        plt.title('LSH Result')
        plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No LSH Match', ha='center', va='center', fontsize=12)
        plt.axis('off')

    plt.suptitle('Comparison of Brute Force vs LSH')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"可视化结果已保存: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    folder = "img/dataset"
    image_paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', 'jfif'))
    ]
    image_paths.sort()

    features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"无法读取图片: {path}")
            continue
        feat = compute_color_histogram(img)
        features.append(feat)
    data = np.vstack(features)

    # 读取目标图片
    target_path = os.path.join(os.getcwd(), "img/target.jpg")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"未找到目标图片: {target_path}")
    query_img = cv2.imread(target_path)
    query_feat = compute_color_histogram(query_img)

    hash_tables, index_sets = build_lsh_table(data, C=3, L=8, m=6)
    nn_time = 0
    lsh_time = 0
    # 检索
    for i in range(0, 10000):
        # 暴力 NN
        start = time()
        nn_result_idx = brute_force_search(query_feat, data)
        nn_time += time() - start

        # LSH
        start = time()
        lsh_result_idx = lsh_search(query_feat, data, hash_tables, index_sets, C=3)
        lsh_time += time() - start

    # 路径
    lsh_img_path = image_paths[lsh_result_idx] if lsh_result_idx is not None else None
    nn_img_path = image_paths[nn_result_idx]

    print("=== 检索结果 ===")
    print("目标图像:", os.path.basename(target_path))
    print("暴力 NN 检索结果:", os.path.basename(nn_img_path))
    if lsh_img_path:
        print("LSH 检索结果:", os.path.basename(lsh_img_path))
    else:
        print("LSH 未找到候选")
    print("=== 时间对比 ===")
    print(f"暴力 NN 搜索时间: {nn_time / 10:.8f} ms")
    print(f"LSH 搜索时间: {lsh_time / 10:.8f} ms")

    # 显示结果
    save_path = "result_compare.png"
    visualize_results(target_path, nn_img_path, lsh_img_path, save_path)