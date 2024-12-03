import os
from glob import glob
from PIL import Image
import numpy as np
import faiss

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from langchain_community.embeddings.bedrock import BedrockEmbeddings
import base64
from PIL import Image
from io import BytesIO
import json
import boto3
import csv









def visualize_results(query, retrieved_images, retrieved_metadata):
    # 한글 폰트 설정
    # plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS의 경우
    # plt.rcParams['font.family'] = 'NanumGothic'  # Linux의 경우

    plt.figure(figsize=(12, 5))

    # If image query
    if isinstance(query, Image.Image):
        plt.subplot(1, len(retrieved_images) + 1, 1)
        plt.imshow(query)
        plt.title("Query Image")
        plt.axis('off')
        start_idx = 2

    # If text query
    else:
        plt.subplot(1, len(retrieved_images) + 1, 1)
        plt.text(0.5, 0.5, f"Query:\n\n '{query}'", fontsize=16, ha='center', va='center')
        plt.axis('off')
        start_idx = 2

    # Display images and metadata
    for i, (img_path, meta) in enumerate(zip(retrieved_images, retrieved_metadata)):
        plt.subplot(1, len(retrieved_images) + 1, i + start_idx)
        plt.imshow(Image.open(img_path))
        # Convert metadata to a pretty-printed JSON string
        meta_str = json.dumps(meta, indent=2, ensure_ascii=False)
        plt.title(f"Match {i + 1}\n{meta_str}")
        plt.axis('off')

    plt.show()



# Initialize BedrockEmbeddings model
bedrock_model = BedrockEmbeddings(credentials_profile_name="default",
                model_id="amazon.titan-embed-image-v1")

# bedrock = boto3.client(service_name="bedrock")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")




def get_metadata_by_product_code(file_path, product_code):
    """
    Reads a CSV file and returns the rows where the product_code matches.

    :param file_path: Path to the CSV file.
    :param product_code: The product code to search for.
    :return: A list of dictionaries representing the rows with the matching product code.
    """
    matching_rows = []
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            if row.get('product_code') == product_code:
                matching_rows.append(row)
    
    return matching_rows

# Example usage
file_path = 'meta_dataset/product_meta.csv'


# print(get_metadata_by_product_code(file_path,"8801068927931"))





# Function to generate embeddings for images using BedrockEmbeddings
def generate_image_embeddings(images_path):
    image_paths = glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
    embeddings = []


    for img_path in image_paths:
        image = Image.open(img_path)
        # 지원되는 최대 이미지 크기는 2048 x 2048 픽셀입니다
        with open(img_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode('utf8')

        # Extract the part of the filename before the underscore
        filename = os.path.basename(img_path)
        filename_without_ext = os.path.splitext(filename)[0]  # Remove the file extension
        parts = filename_without_ext.split('_')
        product_code = parts[0] if len(parts) > 1 else filename_without_ext
        print(f"generate_image_embeddings: {product_code}")

        # 텍스트나 이미지 또는 둘 다 지정할 수 있습니다
        body = json.dumps(
            {
                "inputImage": input_image
            }
        )

        response = bedrock_runtime.invoke_model(
            body=body, 
            modelId="amazon.titan-embed-image-v1", 
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        print(response_body.get("embedding"))
        embeddings.append(response_body.get("embedding"))


    return embeddings, image_paths

# Generate image embeddings
IMAGES_PATH = 'image_dataset'




def create_or_load_faiss_index(output_path, metadata_path, metas_path):
    """
    Create a new FAISS index or load an existing one from the specified path using langchain's FAISS.

    :param output_path: Path to save/load the FAISS index.
    :param metadata_path: Path to save/load the metadata.
    :param metas_path: Path to save/load the metas from product_meta.csv.
    :return: FAISS index object.
    """
    metadata = []
    image_paths = []
    if os.path.exists(output_path) and os.path.exists(metadata_path) and os.path.exists(metas_path):
        # Load existing index
        index = faiss.read_index(output_path)
        print(f"Index loaded from {output_path}")
        
        # Load metadata from CSV
        
        with open(metadata_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_paths.append(row)
        print(f"Metadata loaded from {metadata_path}")
        
        # Load metas
        with open(metas_path, 'r', encoding='utf-8') as f:
            if os.path.getsize(metas_path) > 0:  # Check if file is not empty
                metas = json.load(f)
                print(f"Metas loaded from {metas_path}")
            else:
                metas = []
                print(f"Metas file {metas_path} is empty.")
    else:
        image_embeddings, image_paths = generate_image_embeddings(IMAGES_PATH)
        dimension = len(image_embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIDMap(index)
        
        vectors = np.array(image_embeddings).astype(np.float32)

        # Add vectors to the index with IDs
        index.add_with_ids(vectors, np.array(range(len(image_embeddings))))
        
        # Save the index
        faiss.write_index(index, output_path)
        print(f"Index created and saved to {output_path}")
        
        # Save image paths with UTF-8 encoding
        with open(output_path + '.paths', 'w', encoding='utf-8') as f:
            for img_path in image_paths:
                f.write(img_path + '\n')
        
        
        # Save metas from product_meta.csv with UTF-8 encoding
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            filename_without_ext = os.path.splitext(filename)[0]  # Remove the file extension
            parts = filename_without_ext.split('_')
            product_code = parts[0] if len(parts) > 1 else filename_without_ext
            print(f"create_or_load_faiss_index: {product_code}")
            meta_data = get_metadata_by_product_code(file_path, product_code)
            if meta_data:
                # Ensure UTF-8 encoding
                utf8_meta_data = {k: v for k, v in meta_data[0].items()}
                metadata.append(utf8_meta_data)
            else:
                metadata.append({})

        # Save metas from product_meta.csv with UTF-8 encoding
        with open(metas_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Metas saved to {metas_path}")
        
    return index, metadata, image_paths

# Update the function call to include the new metas_path parameter
image_index, image_metadata_list, image_metas_list = create_or_load_faiss_index("image_vector.index", "image_vector.index.paths", "image_vector_index_metas.json")



def image_embedding(img_path):
    with open(img_path, "rb") as image_file:
        input_image = base64.b64encode(image_file.read()).decode('utf8')

    # Prepare the request body
    body = json.dumps({"inputImage": input_image})

    # Invoke the model to get the embedding
    response = bedrock_runtime.invoke_model(
        body=body, 
        modelId="amazon.titan-embed-image-v1", 
        accept="application/json", 
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    # print(response_body.get("embedding"))

    return response_body.get("embedding")



# Function to retrieve similar images
def retrieve_similar_images(query_image_path, index, top_k=3, distance_threshold=0.6):
    # Load image paths from the file
    with open("image_vector.index.paths", 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f]

    # Load metadata from the metas file
    with open("image_vector_index_metas.json", 'r', encoding='utf-8') as f:
        all_metadata = json.load(f)

    # Generate the embedding for the query image
    query_features = image_embedding(query_image_path)
    
    # Convert the list to a NumPy array and reshape it
    query_features = np.array(query_features).astype(np.float32).reshape(1, -1)

    # Perform the search
    distances, indices = index.search(query_features, top_k)
    print(indices)
    print(distances)

    # Filter results with distances >= 0.6
    filtered_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold]

    # Retrieve images and metadata
    retrieved_images = [image_paths[int(idx)] for idx, _ in filtered_results]
    retrieved_metadata = [all_metadata[int(idx)] for idx, _ in filtered_results]

    return query_image_path, retrieved_images, retrieved_metadata




# Example usage for image similarity
query_image_path = 'image_dataset/pexels-w-w-299285-889839.jpg'
query_image_path = 'test2.jpg'

quert_image_path, retrieved_images, retrieved_metadata = retrieve_similar_images(query_image_path, image_index, top_k=3, distance_threshold=0.6)
print(retrieved_metadata)
visualize_results(quert_image_path, retrieved_images, retrieved_metadata)


