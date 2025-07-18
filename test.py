import os
import shutil

src_root = "/data2/elo/khoatn/clip_qwenvl2.5_0.05"
dst_root = "/data2/elo/khoatn/Multi-Image-Attack-in-RAG/attack_result_usingquestion=1/clip_qwenvl2.5_0.05"

files_to_copy = [
    "adv_4.pkl",
    "adv_history_4.pkl",
    "answers_4.json",
    "images_4.pkl",
    "individuals_4.pkl",
    "scores_4.pkl",
]

for sample_id in os.listdir(src_root):
    src_sample_dir = os.path.join(src_root, sample_id)
    dst_sample_dir = os.path.join(dst_root, sample_id)

    if not os.path.exists(src_sample_dir):
        print(f"Source directory does not exist for sample_id: {sample_id}")
        continue

    copied_any = False
    for fname in files_to_copy:
        src_file = os.path.join(src_sample_dir, fname)
        if os.path.exists(src_file):
            os.makedirs(dst_sample_dir, exist_ok=True)
            shutil.copy2(src_file, dst_sample_dir)
            print(f"Copied {fname} for sample_id: {sample_id}")
            copied_any = True

    if not copied_any:
        print(f"No target files found for sample_id: {sample_id}")
