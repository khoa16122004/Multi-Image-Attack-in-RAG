import json
import os
from retriever import Retriever
anno_path = "v2_anno.jsonl"
run_path = "run.txt"

retriever = Retriever(model_name='clip')
# retri bằng cách tính simiarty giữa text và ảnh
# retriever(qs, imgs_files) # img_files là list của các PILLOw image
with open(run_path, "r") as f:
    # lines contain sample ids
    lines = [int(line.strip()) for line in f.readlines()]
    

for i, line in enumerate(open(anno_path, 'r')):
    # read line từng dòng nếu là sample id thì hãy xử lý
    data = json.loads(line)
    images = data['images']
    question = data['question']
    
    hit_rate = 0
    recall = 0
    mrr = 0
    for file_path, is_gt in data.items():
        print(file_path)
        print(type(is_gt))
        break
    
    break
