import os
import json
from pycocotools.coco import COCO

def prepare_splits(
    coco_json_path="/kaggle/input/datasets/nta212/ip102-for-object-detection/train.json",
    voc_img_dir="/kaggle/input/datasets/nta212/ip102-for-object-detection/VOC2007/JPEGImages",
    output_dir="data/IP102/VOC2007/ImageSets/Main",
    num_exemplars_per_class=20
):
    """
    Chuẩn bị dữ liệu ImageSets cho IP102 dataset (25 lớp, chia làm 4 task):
    - Tự động kiểm tra file ảnh thực tế trên đĩa để tránh lỗi FileNotFoundError.
    """
    print(f"Đang đọc file annotations: {coco_json_path}")
    coco = COCO(coco_json_path)

    # Đọc danh sách file ảnh thực tế đang có trên đĩa Kaggle để đối chiếu 100%
    existing_stems = set()
    if os.path.exists(voc_img_dir):
        for f in os.listdir(voc_img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                existing_stems.add(os.path.splitext(f)[0])
        print(f"✓ Đã quét {len(existing_stems)} file ảnh thực tế trong {voc_img_dir}")
    else:
        print(f"⚠️ Thư mục ảnh {voc_img_dir} chưa sẵn sàng, sẽ dùng fallback format.")

    def get_valid_stem(img_id):
        img_info = coco.imgs.get(img_id, {})
        file_name = img_info.get("file_name", "")
        fn_stem = os.path.splitext(os.path.basename(file_name))[0] if file_name else ""
        
        # Danh sách các ứng viên tên file
        candidates = []
        if str(img_id).isdigit():
            val = int(img_id)
            candidates.append(f"{val:06d}")
            candidates.append(f"{val % 1000000:06d}")
            candidates.append(str(val))
        if fn_stem:
            candidates.append(fn_stem)

        for cand in candidates:
            if cand in existing_stems:
                return cand
        return candidates[0] if candidates else str(img_id)

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
        valid_stems = [get_valid_stem(img_id) for img_id in img_ids]
        with open(path, "w") as f:
            for stem in valid_stems:
                f.write(f"{stem}\n")
        print(f"Đã tạo {name}.txt với {len(valid_stems)} ảnh. Ví dụ mẫu: {valid_stems[:3]}")

if __name__ == "__main__":
    prepare_splits()
