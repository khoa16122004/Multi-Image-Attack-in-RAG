import os
import json

def generate_filter_set_and_average(clean_dir, output_dir, max_topk=5, mode="write"):
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(clean_dir)
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for k in range(1, max_topk + 1):
        set_file_path = os.path.join(model_output_dir, f"set_top{k}.txt")
        average_score_file_path = os.path.join(model_output_dir, f"average_score_top{k}.txt")
        
        total_score = 0
        valid_count = 0

        filtered_sample_ids = []  # Store sample IDs with score > 0

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
                        filtered_sample_ids.append(sample_id)
                        total_score += score
                        valid_count += 1

        # Calculate average score
        average_score = total_score / valid_count if valid_count > 0 else 0

        if mode == "write":
            # Write filtered sample IDs to set_top{k}.txt
            with open(set_file_path, "w") as set_file:
                for sample_id in filtered_sample_ids:
                    set_file.write(f"{sample_id}\n")
            
            # Write average score to average_score_top{k}.txt
            with open(average_score_file_path, "w") as avg_file:
                avg_file.write(f"Average Score for Top{k}: {average_score:.4f}\n")
            
            print(f"Filter set and average score for top{k} written to {model_output_dir}")
        else:
            # Print results without writing to files
            print(f"Top{k}: Average Score = {average_score:.4f}, Valid Samples = {valid_count}")

# Run the function
clean_dir = "clean_result/deepseek-vl2-tiny"  # Replace with your actual clean_dir path
output_dir = "filter_set"
mode = "print"  # Change to "calculate" if you only want to calculate without writing
generate_filter_set_and_average(clean_dir, output_dir, mode=mode)