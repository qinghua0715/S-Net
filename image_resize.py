import imageio
import numpy as np
import os
from PIL import Image

# 定义输入和输出路径
file_path = r'F:\python\data\ISIC2018\ISIC2018_256'
save_path = r'F:\python\data\ISIC2018\ISIC2018_png_224'

# 检查输入路径是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Input path {file_path} does not exist.")

# 定义输入和输出子目录
input_image_dir = os.path.join(file_path, 'image')
input_gt_dir = os.path.join(file_path, 'gt')
output_image_dir = os.path.join(save_path, 'image')
output_gt_dir = os.path.join(save_path, 'label')

# 检查输入子目录是否存在
if not os.path.exists(input_image_dir):
    raise FileNotFoundError(f"Input image directory {input_image_dir} does not exist.")
if not os.path.exists(input_gt_dir):
    raise FileNotFoundError(f"Input ground truth directory {input_gt_dir} does not exist.")

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_gt_dir, exist_ok=True)

# 获取文件列表
image_name_list = os.listdir(input_image_dir)
label_name_list = os.listdir(input_gt_dir)

# 处理图像
for name in image_name_list:
    if name.endswith(('.jpg', '.png')):
        # 读取图像
        image_path = os.path.join(input_image_dir, name)
        try:
            img = imageio.imread(image_path)
        except Exception as e:
            print(f"Failed to read image {image_path}: {e}")
            continue

        # 调整图像尺寸
        img_pil = Image.fromarray(img).resize((224, 224), Image.LANCZOS)
        img_resized = np.array(img_pil)

        # 保存图像
        output_path = os.path.join(output_image_dir, os.path.splitext(name)[0] + '.png')
        imageio.imwrite(output_path, img_resized)
        print(f"Processed image: {name}")

# 处理标签
for name in label_name_list:
    if name.endswith(('.jpg', '.png')):
        # 读取标签
        label_path = os.path.join(input_gt_dir, name)
        try:
            label = imageio.imread(label_path)
        except Exception as e:
            print(f"Failed to read label {label_path}: {e}")
            continue

        # 转换为灰度图并调整尺寸
        label_pil = Image.fromarray(label, mode='L').resize((224, 224), Image.NEAREST)
        label_array = np.array(label_pil)

        # 确保标签是黑白二值图
        label_binary = (label_array > 128).astype(np.uint8) * 255

        # 保存标签
        output_path = os.path.join(output_gt_dir, os.path.splitext(name)[0] + '.png')
        imageio.imwrite(output_path, label_binary)
        print(f"Processed label: {name}")

print("All images and labels have been resized and saved as PNG.")