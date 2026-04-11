import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import json
import os 
import sys 
from pathlib import Path # 引入Path，方便处理路径
import glob # 引入glob，方便查找图片

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np


def load_class_name_map(cfg):
    dataset_cfg = cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {})
    if not dataset_cfg:
        dataset_cfg = cfg.yaml_cfg.get('train_dataloader', {}).get('dataset', {})

    ann_file = dataset_cfg.get('ann_file')
    if ann_file and os.path.exists(ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        categories = data.get('categories', [])
        if categories:
            return {int(cat['id']): str(cat['name']) for cat in categories}

    num_classes = int(cfg.yaml_cfg.get('num_classes', 0) or 0)
    return {i: f'class_{i}' for i in range(num_classes)}

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    # ... (这部分函数保持不变) ...
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - y3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]

def slice_image(image, slice_height, slice_width, overlap_ratio):
    # ... (这部分函数保持不变) ...
    img_width, img_height = image.size
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates

def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    # ... (这部分函数保持不变) ...
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)

def draw(images, labels, boxes, scores, class_name_map, thrh = 0.6, path = ""):
    # Load font once
    try:
        font_obj = ImageFont.load_default(size=12)
    except TypeError:
        font_obj = ImageFont.load_default()

    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red', width=3)
            
            class_id = lab[j].item()
            class_name = class_name_map.get(int(class_id), f"Unknown:{class_id}")
            text = f"{class_name} {round(scrs[j].item(), 2)}"
            
            # Correctly get text width as an integer
            try:
                text_width = font_obj.getlength(text)
            except AttributeError:
                text_width, _ = font_obj.getsize(text)
            
            # Convert tensor coordinates to floats before calculation
            x1, y1 = b[0].item(), b[1].item()
            
            draw.rectangle([x1, y1 - 15, x1 + text_width + 6, y1], fill='red')
            draw.text((x1 + 3, y1 - 15), text=text, font=font_obj, fill='white')

        # This is the solution for problems one and two
        if path == "":
            # If no path is specified, save in the logs directory by default
            output_dir = Path("logs")
            output_dir.mkdir(exist_ok=True)
            im.save(output_dir / f'results_{i}.jpg')
        else:
            # If a path is specified, save to the specified path
            im.save(path)
            
def main(args, ):
    cfg = YAMLConfig(args.config, resume=args.resume)
    class_name_map = load_class_name_map(cfg)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        state = checkpoint.get('ema', {}).get('module', checkpoint['model'])
    else:
        raise AttributeError('必须提供模型权重文件 (-r 参数).')

    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    model.eval() # 确保模型在评估模式

    # =================================================================
    #  问题一和二的解决方案：处理单个文件或整个文件夹
    # =================================================================
    input_path = Path(args.im_file)
    image_paths = []
    if input_path.is_dir():
        # 使用 pathlib.Path.glob 来获取 Path 对象列表，而不是字符串列表
        image_paths.extend(input_path.glob("*.jpg"))
        image_paths.extend(input_path.glob("*.png"))
        image_paths.extend(input_path.glob("*.jpeg"))
        # 将生成器转换为列表以获取长度
        image_paths = list(image_paths)
        print(f"找到了 {len(image_paths)} 张图片在文件夹 {input_path} 中。")
    elif input_path.is_file():
        image_paths.append(input_path)
        print(f"正在处理单张图片: {input_path}")
    else:
        print(f"错误: 路径 {input_path} 不是一个有效的文件或文件夹。")
        return

    # 创建保存结果的目录
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    print(f"结果将保存在: {output_dir}")

    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])

    # Add timing logic for FPS calculation
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    total_time = 0

    # Warm-up runs
    print("Warming up the model...")
    for _ in range(10):
        if image_paths:
            im_data = transforms(Image.open(image_paths[0]).convert('RGB'))[None].to(args.device)
            with torch.no_grad():
                _ = model(im_data, torch.tensor([[im_data.shape[-2], im_data.shape[-1]]]).to(args.device))

    print("Starting inference and FPS measurement...")
    for image_path in image_paths:
        print(f"--- 正在预测: {image_path.name} ---")
        im_pil = Image.open(image_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        
        im_data = transforms(im_pil)[None].to(args.device)
        
        start_time.record()
        with torch.no_grad(): # 在推理时关闭梯度计算，节省显存和时间
            if args.sliced:
                # ... (切片逻辑保持不变) ...
                pass
            else:
                output = model(im_data, orig_size)
                labels, boxes, scores = output
        end_time.record()
        torch.cuda.synchronize() # Wait for the operation to complete
        total_time += start_time.elapsed_time(end_time) # time in milliseconds

        # 生成动态的输出文件名
        output_filename = output_dir / f"{image_path.stem}_result.jpg"
        
        draw([im_pil], labels, boxes, scores, class_name_map, 0.4, path=output_filename) # 降低了阈值以显示更多结果
        print(f"结果已保存至: {output_filename}")

    if total_time > 0:
        fps = len(image_paths) / (total_time / 1000) # Convert ms to s
        print("=================================================================")
        print(f"所有图片处理完毕!")
        print(f"总图片数: {len(image_paths)}")
        print(f"总耗时: {total_time / 1000:.2f} 秒")
        print(f"平均 FPS: {fps:.2f}")
        print("=================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="配置文件路径")
    parser.add_argument('-r', '--resume', type=str, help="模型权重文件路径")
    parser.add_argument('-f', '--im-file', type=str, help="图片文件或文件夹路径")
    parser.add_argument('-s', '--sliced', type=bool, default=False, help="是否使用切片推理")
    parser.add_argument('-d', '--device', type=str, default='cuda', help="推理设备, e.g., 'cpu' or 'cuda'")
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)
