import os

def find_intersection(models_dir, output_dir, max_topk=5):
    os.makedirs(output_dir, exist_ok=True)

    for k in range(1, max_topk + 1):
        intersection_set = None
        for model_name in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_name)
            if not os.path.isdir(model_path):
                continue
            
            set_file_path = os.path.join(model_path, f"set_top{k}.txt")
            if os.path.exists(set_file_path):
                with open(set_file_path, "r") as f:
                    sample_ids = set(line.strip() for line in f)
                    if intersection_set is None:
                        intersection_set = sample_ids
                    else:
                        intersection_set &= sample_ids  # Find intersection
        
        # Save the intersection results to a file
        if intersection_set is not None:
            intersection_file_path = os.path.join(output_dir, f"intersection_top{k}.txt")
            with open(intersection_file_path, "w") as f:
                for sample_id in sorted(intersection_set):
                    f.write(f"{sample_id}\n")
            print(f"Intersection for top{k} saved to {intersection_file_path}")

# Run the function
models_dir = "filter_set"  # Replace with the directory containing model subdirectories
output_dir = "filter_set/intersection"
find_intersection(models_dir, output_dir)