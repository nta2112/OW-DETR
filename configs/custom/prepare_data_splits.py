import os
import json
from pycocotools.coco import COCO

def prepare_splits(
    coco_json_path="/kaggle/input/datasets/nta212/ip102-for-object-detection/train.json",
    output_dir="data/IP102/VOC2007/ImageSets/Main",
    num_exemplars_per_class=20
):
    """
    Chuẩn bị dữ liệu ImageSets cho IP102 dataset (25 lớp, chia làm 4 task):
    - Task 1: 7 lớp (chỉ số 0..6)
    - Task 2: 6 lớp (chỉ số 7..12)
    - Task 3: 6 lớp (chỉ số 13..18)
    - Task 4: 6 lớp (chỉ số 19..24)
    """
    print(f"Đang đọc file annotations: {coco_json_path}")
    coco = COCO(coco_json_path)

    # Ánh xạ danh sách categories được sắp xếp theo ID sang chỉ số 0..24
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c['id'])
    
    # Chia ID danh mục của 25 lớp vào 4 task
    task_cat_ids = {
        't1': set([c['id'] for c in cats[0:7]]),
        't2': set([c['id'] for c in cats[7:13]]),
        't3': set([c['id'] for c in cats[13:19]]),
        't4': set([c['id'] for c in cats[19:25]])
    }

    # 1. Tạo tập train cho từng task dựa trên ID danh mục thực tế
    train_images = {}
    for task_name, cat_ids in task_cat_ids.items():
        task_img_ids = set()
        for ann in coco.anns.values():
            if ann['category_id'] in cat_ids:
                task_img_ids.add(ann['image_id'])
        train_images[task_name] = sorted(list(task_img_ids))

    # Hàm lấy mẫu đại diện (exemplars) từ các lớp cũ để finetuning
    def get_exemplars(cat_ids_set, num_images_per_class=num_exemplars_per_class):
        exemplar_ids = set()
        for cid in cat_ids_set:
            img_ids = [ann['image_id'] for ann in coco.anns.values() if ann['category_id'] == cid]
            selected = img_ids[:num_images_per_class]
            exemplar_ids.update(selected)
        return exemplar_ids

    # 2. Tạo tập Finetuning (ft) giữ lại tri thức các task trước
    ft_images = {
        't2': sorted(list(set(train_images['t2']).union(get_exemplars(task_cat_ids['t1'])))),
        't3': sorted(list(set(train_images['t3']).union(get_exemplars(task_cat_ids['t1'] | task_cat_ids['t2'])))),
        't4': sorted(list(set(train_images['t4']).union(get_exemplars(task_cat_ids['t1'] | task_cat_ids['t2'] | task_cat_ids['t3']))))
    }

    all_img_ids = sorted(list(coco.imgs.keys()))
    os.makedirs(output_dir, exist_ok=True)

    splits = {
        't1_train': train_images['t1'],
        't2_train': train_images['t2'],
        't3_train': train_images['t3'],
        't4_train': train_images['t4'],
        't2_ft': ft_images['t2'],
        't3_ft': ft_images['t3'],
        't4_ft': ft_images['t4'],
        'val': all_img_ids,
        'test': all_img_ids
    }

    for name, img_ids in splits.items():
        path = os.path.join(output_dir, f"{name}.txt")
        with open(path, "w") as f:
            for img_id in img_ids:
                # Định dạng tên file gồm 6 chữ số số nguyên (vd: 001536) khớp 100% với file ảnh .jpg và XML trên Kaggle
                formatted_name = f"{int(img_id):06d}" if str(img_id).isdigit() else str(img_id)
                f.write(f"{formatted_name}\n")
        print(f"Đã tạo {name}.txt với {len(img_ids)} ảnh.")

if __name__ == "__main__":
    prepare_splits()
