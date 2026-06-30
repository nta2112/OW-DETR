import os
import shutil
import xml.etree.cElementTree as ET
import argparse
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description="Convert IP102 COCO annotations to Pascal VOC XML format")
    parser.add_argument("--coco_dir", default="/kaggle/input/datasets/eljazouly/ip102-coco-annotations/coco_annotations", type=str, help="Thư mục chứa các file json annotations của COCO")
    parser.add_argument("--img_dir", default="/kaggle/input/datasets/rtlmhjbn/ip02-dataset/classification", type=str, help="Thư mục chứa ảnh gốc phân theo lớp")
    parser.add_argument("--classes_file", default="/kaggle/input/datasets/rtlmhjbn/ip02-dataset/classes.txt", type=str, help="Đường dẫn đến file classes.txt gốc")
    parser.add_argument("--output_dir", default="data/IP102", type=str, help="Thư mục lưu dataset VOC2007 đầu ra")
    return parser.parse_args()

def process_split(coco_dir, img_dir, output_dir, json_name, split_name, images_target_dir, annotations_target_dir, splits_target_dir, classes):
    json_path = os.path.join(coco_dir, json_name)
    if not os.path.exists(json_path):
        print(f"File {json_path} không tồn tại, bỏ qua split này.")
        return
        
    print(f"Đang xử lý split: {split_name} từ {json_name}...")
    coco = COCO(json_path)
    
    split_txt_path = os.path.join(splits_target_dir, f"{split_name}.txt")
    
    # Một số split có thể trùng lặp ảnh nên chúng ta dùng tập hợp để tránh trùng lặp
    processed_ids = set()
    
    with open(split_txt_path, "w") as split_file:
        for index, img_id in enumerate(coco.imgs):
            img_details = coco.imgs[img_id]
            file_name = img_details['file_name']
            
            # Tạo tên chuẩn hóa gồm 6 ký tự số từ ID để khớp với convert_image_id trong code OW-DETR
            formatted_name = f"{img_id:06d}"
            
            # 1. Copy/Link ảnh
            src_image_path = os.path.join(img_dir, file_name)
            dst_image_path = os.path.join(images_target_dir, f"{formatted_name}.jpg")
            
            if os.path.exists(src_image_path):
                if not os.path.exists(dst_image_path):
                    # Sử dụng symlink nếu chạy trên Linux/Kaggle để tiết kiệm dung lượng và thời gian, 
                    # nếu lỗi thì copy
                    try:
                        os.symlink(src_image_path, dst_image_path)
                    except Exception:
                        shutil.copy(src_image_path, dst_image_path)
            else:
                # Tìm kiếm tương đối nếu file_name có đường dẫn khác
                alternative_path = os.path.join(img_dir, os.path.basename(file_name))
                if os.path.exists(alternative_path):
                    if not os.path.exists(dst_image_path):
                        try:
                            os.symlink(alternative_path, dst_image_path)
                        except Exception:
                            shutil.copy(alternative_path, dst_image_path)
                else:
                    print(f"Cảnh báo: Không tìm thấy ảnh nguồn tại {src_image_path} hoặc {alternative_path}")
                    continue
                
            # 2. Tạo file nhãn XML (Pascal VOC format)
            annotation_el = ET.Element('annotation')
            ET.SubElement(annotation_el, 'filename').text = f"{formatted_name}.jpg"
            
            size_el = ET.SubElement(annotation_el, 'size')
            ET.SubElement(size_el, 'width').text = str(img_details['width'])
            ET.SubElement(size_el, 'height').text = str(img_details['height'])
            ET.SubElement(size_el, 'depth').text = "3"
            
            # Đọc bounding box tương ứng
            if img_id in coco.imgToAnns:
                for ann in coco.imgToAnns[img_id]:
                    cat_id = ann['category_id']
                    if cat_id in coco.cats:
                        cat_name = coco.cats[cat_id]['name']
                    elif (cat_id + 1) in coco.cats:
                        cat_name = coco.cats[cat_id + 1]['name']
                    elif len(classes) > 0:
                        if 0 <= cat_id < len(classes):
                            cat_name = classes[cat_id]
                        elif 0 <= cat_id - 1 < len(classes):
                            cat_name = classes[cat_id - 1]
                        else:
                            cat_name = f"class_{cat_id}"
                    else:
                        cat_name = f"class_{cat_id}"
                    
                    object_el = ET.SubElement(annotation_el, 'object')
                    ET.SubElement(object_el, 'name').text = cat_name
                    ET.SubElement(object_el, 'difficult').text = '0'
                    
                    bbox = ann['bbox']
                    # COCO bbox: [xmin, ymin, width, height]
                    # VOC bbox: [xmin, ymin, xmax, ymax]
                    xmin = int(bbox[0] + 1)
                    ymin = int(bbox[1] + 1)
                    xmax = int(bbox[0] + bbox[2] + 1)
                    ymax = int(bbox[1] + bbox[3] + 1)
                    
                    bb_el = ET.SubElement(object_el, 'bndbox')
                    ET.SubElement(bb_el, 'xmin').text = str(xmin)
                    ET.SubElement(bb_el, 'ymin').text = str(ymin)
                    ET.SubElement(bb_el, 'xmax').text = str(xmax)
                    ET.SubElement(bb_el, 'ymax').text = str(ymax)
            
            xml_path = os.path.join(annotations_target_dir, f"{formatted_name}.xml")
            ET.ElementTree(annotation_el).write(xml_path)
            
            # 3. Ghi id vào split txt
            split_file.write(f"{formatted_name}\n")
            
            if (index + 1) % 2000 == 0:
                print(f"Đã xử lý {index + 1} ảnh...")
                
    print(f"Đã hoàn thành split: {split_name}.")

def main():
    args = parse_args()
    
    voc_dir = os.path.join(args.output_dir, "VOC2007")
    images_target_dir = os.path.join(voc_dir, "JPEGImages")
    annotations_target_dir = os.path.join(voc_dir, "Annotations")
    splits_target_dir = os.path.join(voc_dir, "ImageSets/Main")
    
    os.makedirs(images_target_dir, exist_ok=True)
    os.makedirs(annotations_target_dir, exist_ok=True)
    os.makedirs(splits_target_dir, exist_ok=True)
    
    # Nạp danh sách lớp từ file classes.txt để dự phòng ánh xạ
    classes = []
    if os.path.exists(args.classes_file):
        with open(args.classes_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) > 1 and parts[0].isdigit():
                    classes.append(parts[1])
                else:
                    classes.append(line)
        shutil.copy(args.classes_file, os.path.join(voc_dir, "classes.txt"))
        print(f"Đã copy {args.classes_file} sang {voc_dir}/classes.txt")
    else:
        print(f"Cảnh báo: Không thấy file classes.txt tại {args.classes_file}. Hãy tự chuẩn bị sau.")

    process_split(args.coco_dir, args.img_dir, args.output_dir, "train.json", "train", images_target_dir, annotations_target_dir, splits_target_dir, classes)
    process_split(args.coco_dir, args.img_dir, args.output_dir, "val.json", "val", images_target_dir, annotations_target_dir, splits_target_dir, classes)
    process_split(args.coco_dir, args.img_dir, args.output_dir, "test.json", "test", images_target_dir, annotations_target_dir, splits_target_dir, classes)
    
    print("\nQuá trình chuyển đổi định dạng hoàn tất thành công!")
    print(f"Dữ liệu VOC hiện được lưu tại: {voc_dir}")

if __name__ == '__main__':
    main()
