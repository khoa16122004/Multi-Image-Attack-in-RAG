import os

def find_global_intersection(models_dir, output_dir, max_topk=5):
    os.makedirs(output_dir, exist_ok=True)

    global_intersection = None
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        model_intersection = None
        for k in range(1, max_topk + 1):
            set_file_path = os.path.join(model_path, f"set_top{k}.txt")
            if os.path.exists(set_file_path):
                with open(set_file_path, "r") as f:
                    sample_ids = set(line.strip() for line in f)
                    if model_intersection is None:
                        model_intersection = sample_ids
                    else:
                        model_intersection &= sample_ids  # Find intersection for this model
        
        if model_intersection is not None:
            if global_intersection is None:
                global_intersection = model_intersection
            else:
                global_intersection &= model_intersection  # Find intersection across all models

    # Save the global intersection results to a file
    if global_intersection is not None:
        global_intersection_file = os.path.join(output_dir, "intersection_all_topk.txt")
        with open(global_intersection_file, "w") as f:
            for sample_id in sorted(global_intersection):
                f.write(f"{sample_id}\n")
        print(f"Global intersection across all topk saved to {global_intersection_file}")
    else:
        print("No intersection found across all models and topk.")

# Run the function
models_dir = "filter_set"  # Replace with the directory containing model subdirectories
output_dir = "filter_set/global_intersection"
find_global_intersection(models_dir, output_dir)