import os
import numpy as np
import matplotlib.pyplot as plt
from extract_feature_resnet50 import extract_features_resnet50
from extract_feature_vgg19 import extract_features_vgg19
from extract_feature_alexnet import extract_features_alexnet

# Extract features of all images in the dataset using ResNet50
def extract_dataset_feature_resnet50(dataset_path, save_path):
    for i in range(1, 53):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_resnet50(input_image_path, save_path_i)


# Extract features of all images in the dataset using VGG16
def extract_dataset_feature_vgg19(dataset_path, save_path):
    for i in range(1, 53):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_vgg19(input_image_path, save_path_i)

# Extract features of all images in the dataset using AlexNet
def extract_dataset_feature_alexnet(dataset_path, save_path):
    for i in range(1, 53):
        input_image_path = dataset_path + f'/{i}.jpg'
        save_path_i = save_path + f'/{i}.npy'
        extract_features_alexnet(input_image_path, save_path_i)

def cal_similarity(feature1, feature2):
    feature1 = feature1.flatten()
    feature2 = feature2.flatten()
    # 归一化
    feature1 = feature1 / np.linalg.norm(feature1)
    feature2 = feature2 / np.linalg.norm(feature2)
    # 欧氏距离
    distance = np.linalg.norm(feature1 - feature2)
    return distance

def query_k_image(query_image_feature_path, dataset_path, k=5):
    query_feature = np.load(query_image_feature_path)
    distances = []
    indices = []
    for i in range(1, 53):
        feature_path = f'{dataset_path}/{i}.npy'
        try:
            feature = np.load(feature_path)
        except FileNotFoundError:
            print(f'Feature {i} not found!')
            continue
        dist = cal_similarity(query_feature, feature)
        distances.append(dist)
        indices.append(i)
    distances = np.array(distances)
    indices = np.array(indices)
    # 从小到大排序
    sorted_indices = np.argsort(distances)
    topk_index = indices[sorted_indices[:k]]
    topk_distance = distances[sorted_indices[:k]]

    return topk_index, topk_distance

def draw(output_path, query_image_path, model, data, i):
    plt.figure(figsize=(28, 5))
    plt.subplot(1, 6, 1)
    query_image = plt.imread(query_image_path)
    plt.imshow(query_image)
    plt.axis('off')
    plt.title('Query Image')
    for j, (idx, sim) in enumerate(data):
        print('Index:', idx, 'Similarity:', sim)
        image_path = f'./Dataset/{idx}.jpg'
        image = plt.imread(image_path)
        plt.subplot(1, 6, j + 2)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'Top {j + 1} similar: Index: {idx}, Similarity: {sim:.2f}')
        plt.axis('off')
        plt.suptitle(f'{model}')
        plt.tight_layout()
        plt.savefig(f'{output_path}/{model}_{i}.png')

def main():
    os.makedirs('./Feature_resnet50', exist_ok=True)
    os.makedirs('./Feature_vgg19', exist_ok=True)
    os.makedirs('./Feature_alexnet', exist_ok=True)
    os.makedirs('./Query_feature', exist_ok=True)
    os.makedirs('./Output', exist_ok=True)

    dataset_path = './Dataset'
    save_path_resnet50 = './Feature_resnet50'
    save_path_vgg19 = './Feature_vgg19'
    save_path_alexnet = './Feature_alexnet'
    output_path = './Output'

    # If the features of the dataset have been extracted, comment the following three lines
    extract_dataset_feature_resnet50(dataset_path, save_path_resnet50)
    extract_dataset_feature_vgg19(dataset_path, save_path_vgg19)
    extract_dataset_feature_alexnet(dataset_path, save_path_alexnet)

    query_images = ['./1.jpg', './2.jpg', './3.jpg']
    for i, query_image_path in enumerate(query_images):
        query_image_path = f'./Query_image/{i+1}.jpg'

        print('Query the most similar image using resnet50!')
        # Extract features of the query image using resnet50
        extract_features_resnet50(query_image_path, f'./Query_feature/query_resnet50_{i}.npy')
        # Query the most similar image using resnet50
        query_image_feature_path = f'./Query_feature/query_resnet50_{i}.npy'
        min_k_index_resnet50, min_k_similarity_resnet50 = query_k_image(query_image_feature_path, './Feature_resnet50', k=5)

        print('Query the most similar image using vgg19!')
        # Extract features of the query image using vgg19
        extract_features_vgg19(query_image_path, f'./Query_feature/query_vgg19_{i}.npy')
        # Query the most similar image using vgg19
        query_image_feature_path = f'./Query_feature/query_vgg19_{i}.npy'
        min_k_index_vgg19, min_k_similarity_vgg19 = query_k_image(query_image_feature_path, './Feature_vgg19', k=5)

        print('Query the most similar image using alexnet!')
        # Extract features of the query image using alexnet
        extract_features_alexnet(query_image_path, f'./Query_feature/query_alexnet_{i}.npy')
        # Query the most similar image using alexnet
        query_image_feature_path = f'./Query_feature/query_alexnet_{i}.npy'
        min_k_index_alexnet, min_k_similarity_alexnet = query_k_image(query_image_feature_path, './Feature_alexnet', k=5)

        print('The query image is:', query_image_path)
        print('The top 5 most similar images using ResNet50:')
        draw(output_path, query_image_path, 'ResNet50', zip(min_k_index_resnet50, min_k_similarity_resnet50), i)
        print('The top 5 most similar images using VGG19:')
        draw(output_path, query_image_path, 'VGG19', zip(min_k_index_vgg19, min_k_similarity_vgg19), i)
        print('The top 5 most similar images using AlexNet:')
        draw(output_path, query_image_path, 'AlexNet', zip(min_k_index_alexnet, min_k_similarity_alexnet), i)

if __name__ == '__main__':
    main()