import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os
from glob import glob
import numpy as np
import faiss
import csv

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# print("Root path:", os.getcwd())
# sys.path.append(os.path.dirname(__file__))
# sys.path.append(os.getcwd())
# print(os.path.dirname(__file__))


import gradio as gr
import json
import numpy as np
import base64
import boto3
from langchain_community.embeddings.bedrock import BedrockEmbeddings


# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


#sam model init#######################################################

# # select the device for computation
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(f"using device: {device}")

# if device.type == "cuda":
#     # use bfloat16 for the entire notebook
#     torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     if torch.cuda.get_device_properties(0).major >= 8:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True

# sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
examples=["test_dataset/product_test_3.jpg","test_dataset/product_test_4.jpg","test_dataset/product_test_5.jpg","test_dataset/candy_sample2.jpg"]
#"test_dataset/product_test_1.jpg",
# sam2 = build_sam2(model_cfg, sam2_checkpoint, device="cpu", apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=16,
#     points_per_batch=4096,
#     pred_iou_thresh=0.9,
#     stability_score_thresh=0.90,
#     stability_score_offset=0.7,
#     crop_n_layers=0,
#     box_nms_thresh=0.7,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=500,
#     use_m2m=True,
# )



IMAGES_PATH = 'image_dataset'
bedrock_model = BedrockEmbeddings(credentials_profile_name="default",
                model_id="amazon.titan-embed-image-v1")
bedrock_runtime = boto3.client(service_name="bedrock-runtime")
file_path = 'meta_dataset/product_meta.csv'
# Initialize BedrockEmbeddings model

########################################################

def show_anns(anns, original_image, borders=True):
    if len(anns) == 0:
        return original_image  # Return the original image if there are no annotations

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.array(original_image.convert("RGBA"))  # Convert original image to RGBA

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = (img[m] * (1 - color_mask[3]) + color_mask * 255).astype(np.uint8)  # Blend the mask with the image
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 255, 255), thickness=1)  # Draw contours in red

    return Image.fromarray(img)


def is_contained(mask1, mask2):
    """
    Check if mask1 is completely contained within mask2.

    :param mask1: First mask.
    :param mask2: Second mask.
    :return: True if mask1 is contained within mask2, False otherwise.
    """
    return np.all(np.logical_and(mask1, mask2) == mask1)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def remove_duplicate_masks(annotations, iou_threshold=0.5, prioritize='size'):
    """
    Remove duplicate masks based on IoU or mask size, and retain larger mask if one is contained within another.

    :param annotations: List of annotations with 'segmentation' masks.
    :param iou_threshold: IoU threshold to consider masks as duplicates.
    :param prioritize: Criteria to prioritize which mask to keep ('iou' or 'size').
    :return: Filtered list of annotations.
    """
    masks = [annotation['segmentation'] for annotation in annotations]
    filtered_annotations = []

    for i, mask1 in enumerate(masks):
        keep_mask = True
        for j, mask2 in enumerate(masks):
            if i != j:
                iou = calculate_iou(mask1, mask2)
                if iou > iou_threshold or is_contained(mask1, mask2):
                    # Prioritize larger mask if one is contained within another
                    if mask1.sum() < mask2.sum():
                        keep_mask = False
                        break
        if keep_mask:
            filtered_annotations.append(annotations[i])
    return filtered_annotations



# bedrock = boto3.client(service_name="bedrock")


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




# Function to generate embeddings for images using BedrockEmbeddings
def generate_image_embeddings(images_path):
    image_paths = glob(os.path.join(images_path, '**/*.jpg'), recursive=True)
    embeddings = []

    for img_path in image_paths:
        with open(img_path, "rb") as image_file:
            # Open the image using PIL
            # input_image = base64.b64encode(image_file.read()).decode('utf8')

            image = Image.open(image_file)
            input_size = 1024
            w, h = image.size
            scale = input_size / max(w, h)
            desired_width = int(w * scale)
            desired_height = int(h * scale)
            image = image.resize((desired_width, desired_height))
            
            # Convert image to RGB if it has an alpha channel
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            input_image = base64.b64encode(buffered.getvalue()).decode('utf8')
        
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


image_index, image_metadata_list, image_metas_list = create_or_load_faiss_index("image_vector.index", "image_vector.index.paths", "image_vector_index_metas.json")

    # Load image paths from the file
with open("image_vector.index.paths", 'r', encoding='utf-8') as f:
    image_paths = [line.strip() for line in f]

# Load metadata from the metas file
with open("image_vector_index_metas.json", 'r', encoding='utf-8') as f:
    all_metadata = json.load(f)



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

def image_embedding_from_array(image_array):
    from io import BytesIO
    from PIL import Image

    # Convert the numpy.ndarray to a PIL Image
    pil_image = Image.fromarray(image_array)

    # Save the PIL Image to a BytesIO object
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    # Encode the image to base64
    input_image = base64.b64encode(buffered.getvalue()).decode('utf8')

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

    return response_body.get("embedding")


def gradio_tab1_function(image):
    
    print(type(image))
    # Convert to PIL Image if it's a NumPy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    input_size = 256  # Define the desired input size
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    nd_image = np.array(image)

    distance_threshold = 0.6
    # distance_threshold = 0
    global image_index, image_paths, all_metadata
    # Generate the embedding for the query image
    query_features = image_embedding_from_array(nd_image)
    
    # Convert the list to a NumPy array and reshape it
    query_features = np.array(query_features).astype(np.float32).reshape(1, -1)

    # Perform the search
    distances, indices = image_index.search(query_features, 1)
    print(indices)
    print(distances)

    # Filter results with distances >= 0.6
    filtered_results = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist >= distance_threshold]

    # Retrieve images and add captions
    retrieved_images_with_captions = [
        (image_paths[int(idx)], f"#{i+1} - 정확도: {dist:.2f}") 
        for i, (idx, dist) in enumerate(filtered_results)
    ]
    retrieved_metadata = [all_metadata[int(idx)] for idx, _ in filtered_results]

    # Convert metadata to pretty JSON format
    if retrieved_metadata==[]:
        retrieved_metadata = [{"need":"상품 사진을 다시 찍어주세요~!","reason":"상품이 중앙에 위치하고, 이미지의 74% 이상 차지하고 있을때 좋은 성능을 보입니다."}]
    pretty_metadata = json.dumps(retrieved_metadata, ensure_ascii=False, indent=2)

    return retrieved_images_with_captions, pretty_metadata

# def gradio_tab2_function(image):
    global mask_generator
    # Ensure the image is a PIL Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    input_size = 1024
    w, h = image.size  # Ensure this is a PIL Image to avoid unpacking error
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    nd_image = np.array(image)
    annotations = mask_generator.generate(nd_image)
    print(len(annotations))
    print(annotations[0].keys())
    print(annotations)
    image=show_anns(annotations, image)
    # Filter out masks that are too large
    
    return image  # Return the processed image

def gradio_tab3_function(image):
    # Process the image for Tab 3
    return image  # Return the processed image

def gradio_tab4_function(image):
    # Process the image for Tab 4
    return image  # Return the processed image

with gr.Blocks(css=".scrollable-gallery { height: 300px; overflow-y: auto; }") as demo:
    gr.Markdown("# GSRETAIL Barcode Free")
    with gr.Tab("상품 스캔 with 이미지"):
        gr.Markdown("# 이미지로 상품 스캔")  # Add title for Tab 1
        with gr.Row():
            with gr.Column(scale=1):
                image_input1 = gr.Image(label="Image Input for Tab 1")
                button1 = gr.Button("Submit")
                clear_button1 = gr.Button("Clear") 
            with gr.Column(scale=2):
                image_output1 = gr.Gallery(label="Image Output for Tab 1", show_label=True, elem_id="scrollable-gallery")  # Apply CSS class
                metadata_output1 = gr.Textbox(label="Image Metadata for Tab 1", lines=5)  # Adjusted for list display
        gr.Examples(
            examples=examples,
            inputs=[image_input1],
            # outputs=segm_img_p,
            # fn=gradio_tab1_function,
            # fn=segment_everything,
            # cache_examples=True,
            examples_per_page=4,
        )
        button1.click(gradio_tab1_function, inputs=[image_input1], outputs=[image_output1, metadata_output1 if metadata_output1 else "사진을 상품이 잘 나오도록 꽉차게 찍어주세요"])
        clear_button1.click(lambda: (None, "", ""), inputs=[], outputs=[image_input1, image_output1, metadata_output1])
        
    # with gr.Tab("Tab 2"):
    #     gr.Markdown("# Tab 2 Title")  # Add title for Tab 2
    #     with gr.Row():
    #         image_input2 = gr.Image(label="Image Input for Tab 2")
    #         image_output2 = gr.Image(label="Image Output for Tab 2")
    #     button2 = gr.Button("Submit")
    #     clear_button2 = gr.Button("Clear")
    #     button2.click(gradio_tab2_function, inputs=[image_input2], outputs=[image_output2])
    #     clear_button2.click(lambda: (None, None), inputs=[], outputs=[image_input2, image_output2])

    # with gr.Tab("Tab 3"):
    #     gr.Markdown("# Tab 3 Title")  # Add title for Tab 3
    #     with gr.Row():
    #         image_input3 = gr.Image(label="Image Input for Tab 3")
    #         image_output3 = gr.Image(label="Image Output for Tab 3")
    #     button3 = gr.Button("Submit")
    #     clear_button3 = gr.Button("Clear")
    #     button3.click(gradio_tab3_function, inputs=[image_input3], outputs=[image_output3])
    #     clear_button3.click(lambda: (None, None), inputs=[], outputs=[image_input3, image_output3])

    # with gr.Tab("Tab 4"):
    #     gr.Markdown("# Tab 4 Title")  # Add title for Tab 4
    #     with gr.Row():
    #         image_input4 = gr.Image(label="Image Input for Tab 4")
    #         image_output4 = gr.Image(label="Image Output for Tab 4")
    #     button4 = gr.Button("Submit")``
    #     clear_button4 = gr.Button("Clear")
    #     button4.click(gradio_tab4_function, inputs=[image_input4], outputs=[image_output4])
    #     clear_button4.click(lambda: (None, None), inputs=[], outputs=[image_input4, image_output4])

# demo.launch(share=True)
demo.queue()
demo.launch(share=True)
