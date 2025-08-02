import os
import json

def generate_filter_set(clean_dir, output_dir, max_topk=5):
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(clean_dir)
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for k in range(1, max_topk + 1):
        set_file_path = os.path.join(model_output_dir, f"set_top{k}.txt")
        with open(set_file_path, "w") as set_file:
            for sample_id in os.listdir(clean_dir):
                sample_path = os.path.join(clean_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                
                answer_path = os.path.join(sample_path, f"answers_top{k}.json")
                if os.path.exists(answer_path):
                    with open(answer_path, "r") as f:
                        data = json.load(f)
                        score = data.get("parse_score", 0)
                        if score > 0:
                            question = data.get("question", "Unknown Question")
                            set_file.write(f"{sample_id}\n")
    
    print(f"Filter set generated in {model_output_dir}")

# Run the function
clean_dir = "clean_result/deepseek-vl2-tiny"  # Replace with your actual clean_dir path
output_dir = "filter_set"
generate_filter_set(clean_dir, output_dir)