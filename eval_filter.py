import os
import json

def calculate_average_score(output_dir):
    average_scores = {}
    for sample_id in os.listdir(output_dir):
        sample_path = os.path.join(output_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue
        
        total_score = 0
        count = 0
        for file_name in os.listdir(sample_path):
            if file_name.startswith("answers_top") and file_name.endswith(".json"):
                file_path = os.path.join(sample_path, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    total_score += data.get("parse_score", 0)
                    count += 1
        
        if count > 0:
            average_scores[sample_id] = total_score / count
    
    # Save the average scores to a JSON file
    average_score_file = os.path.join(output_dir, "average_scores.json")
    with open(average_score_file, "w") as f:
        json.dump(average_scores, f, indent=4)
    print(f"Average scores saved to {average_score_file}")

# Run the function
output_dir = "clean_result"  # Ensure this matches the output directory used in filter.py
calculate_average_score(output_dir)