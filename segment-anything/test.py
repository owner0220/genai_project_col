import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor,  SamAutomaticMaskGenerator
from segment_anything.utils.onnx import SamOnnxModel
from PIL import Image

import onnxruntime
from onnxruntime.quantization import QuantType

from onnxruntime.quantization.quantize import quantize_dynamic

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

#/home/insik_hwang/segment-anything/model_ck/sam_vit_h_4b8939.pth
checkpoint = "./model_ck/sam_vit_h_4b8939.pth"
model_type = "vit_h"


sam = sam_model_registry[model_type](checkpoint=checkpoint)

import warnings

#onnx_model_path = "sam_onnx_example.onnx"

#onnx_model = SamOnnxModel(sam, return_single_mask=True)

#dynamic_axes = {
#    "point_coords": {1: "num_points"},
#    "point_labels": {1: "num_points"},
#}

#embed_dim = sam.prompt_encoder.embed_dim
#embed_size = sam.prompt_encoder.image_embedding_size
#mask_input_size = [4 * x for x in embed_size]
#dummy_inputs = {
#    "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
#    "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
#    "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
#    "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
#    "has_mask_input": torch.tensor([1], dtype=torch.float),
#    "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
#}
#output_names = ["masks", "iou_predictions", "low_res_masks"]

#with warnings.catch_warnings():
#    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
#    warnings.filterwarnings("ignore", category=UserWarning)
#    with open(onnx_model_path, "wb") as f:
#        torch.onnx.export(
#            onnx_model,
#            tuple(dummy_inputs.values()),
#            f,
#            export_params=True,
#            verbose=False,
#            opset_version=17,
#            do_constant_folding=True,
#            input_names=list(dummy_inputs.keys()),
#            output_names=output_names,
#            dynamic_axes=dynamic_axes,
#        )    

def calculate_center(mask):
    # 마스크에서 값이 1인 (y, x) 좌표를 추출
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return (float('inf'), float('inf'))  # 마스크가 없는 경우를 대비한 큰 값 반환
    # y와 x 좌표의 평균을 구해 중심 좌표 계산
    y_mean, x_mean = coords.mean(axis=0)
    return (y_mean, x_mean)



def save_each_masked_area_as_image(image, anns, output_prefix="masked_area"):
    # Read the image with OpenCV (BGR format)
    if image is None:
        print("Error: Unable to read the image.")
        return
    
    
    # Process each annotation separately
    for idx, ann in enumerate(anns):
        m = ann['segmentation']
        
        # Create a transparent base image with only the mask area filled
        mask_shape = m.shape
        img = np.zeros((mask_shape[0], mask_shape[1], 4), dtype=np.uint8)
        
        # Apply mask: copy original image pixels where the mask is true
        img[m, :3] = image[m, :3]   # Copy RGB channels from original
        img[m, 3] = 255                      # Set alpha to opaque
        
        # Find bounding box of the masked area
        rows = np.any(img[:, :, 3], axis=1)
        cols = np.any(img[:, :, 3], axis=0)
        if not np.any(rows) or not np.any(cols):
            print(f"No mask area found for annotation {idx}.")
            continue
        
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        
        # Crop the image to the bounding box
        cropped_img = img[row_min:row_max+1, col_min:col_max+1]
        
        # Convert the cropped numpy array to a PIL Image and save as PNG
        pil_img = Image.fromarray(cropped_img)
        filename = f"{output_prefix}_{idx+1}.png"
        pil_img.save(filename)
        print(f"Masked area {idx+1} saved as {filename}")


def save_each_masked_area_as_image_center(image, anns, output_prefix="masked_area"):
        # 이미지가 None인지 확인
    if image is None:
        print("Error: Unable to read the image.")
        return

    # 각 annotation에 대해 중심 좌표를 계산하고 함께 저장
    masks_with_centers = []
    for idx, ann in enumerate(anns):
        m = ann['segmentation']
        center = calculate_center(m)
        masks_with_centers.append((idx, m, center))

    # 중심 좌표를 기준으로 정렬 (y값, x값 순서)
    masks_with_centers.sort(key=lambda x: (x[2][0], x[2][1]))

    # 정렬된 마스크를 기반으로 이미지를 저장
    for idx, (original_idx, m, center) in enumerate(masks_with_centers):
        # Create a transparent base image with only the mask area filled
        mask_shape = m.shape
        img = np.zeros((mask_shape[0], mask_shape[1], 4), dtype=np.uint8)

        # Apply mask: copy original image pixels where the mask is true
        img[m, :3] = image[m, :3]   # Copy RGB channels from original
        img[m, 3] = 255                      # Set alpha to opaque

        # Find bounding box of the masked area
        rows = np.any(img[:, :, 3], axis=1)
        cols = np.any(img[:, :, 3], axis=0)
        if not np.any(rows) or not np.any(cols):
            print(f"No mask area found for annotation {original_idx}.")
            continue

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Crop the image to the bounding box
        cropped_img = img[row_max:row_min+1, col_max:col_min+1]

        # Convert the cropped numpy array to a PIL Image and save as PNG
        pil_img = Image.fromarray(cropped_img)
        filename = f"{output_prefix}_{idx+1}.png"
        pil_img.save(filename)
        print(f"Masked area {idx+1} (original annotation {original_idx+1}) saved as {filename}")



onnx_model_quantized_path = "sam_onnx_quantized_example.onnx"
#quantize_dynamic(
#    model_input=onnx_model_path,
#    model_output=onnx_model_quantized_path,
#    optimize_model=True,
#    per_channel=False,
#    reduce_range=False,
#    weight_type=QuantType.QUInt8,
#)
onnx_model_path = onnx_model_quantized_path
print("success")
image=cv2.imread('notebooks/images/dog.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


ort_session = onnxruntime.InferenceSession(onnx_model_path)
sam.to(device='cpu')
mask_generator= SamAutomaticMaskGenerator(sam)
mask_generator= SamAutomaticMaskGenerator(model=sam, points_per_side=10,pred_iou_thresh=0.95,stability_score_thresh=0.96,crop_n_layers=1,crop_n_points_downscale_factor=2,min_mask_region_area=30000,)
masks = mask_generator.generate(image)

print(len(masks))
print("###############################")
print(masks)
save_each_masked_area_as_image(image, masks, output_prefix="masked_area")




