import os
import json

def calculate_average_score(output_dir):
    average_scores = {}
    top_scores = {}  # Dictionary to store scores for each top-k level
    for sample_id in os.listdir(output_dir):
        sample_path = os.path.join(output_dir, sample_id)
        if not os.path.isdir(sample_path):
            continue
        
        total_score = 0
        count = 0
        top_scores[sample_id] = []  # Initialize list for top-k scores
        for k in range(1, 6):  # Loop from top1 to top5
            file_name = f"answers_top{k}.json"
            file_path = os.path.join(sample_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    score = data.get("parse_score", 0)
                    top_scores[sample_id].append(score)
                    total_score += score
                    count += 1
        
        if count > 0:
            average_scores[sample_id] = total_score / count
    
    # Save the average scores to a JSON file
    average_score_file = os.path.join(output_dir, "average_scores.json")
    with open(average_score_file, "w") as f:
        json.dump(average_scores, f, indent=4)
    print(f"Average scores saved to {average_score_file}")
    
    # Print the list of scores from top1 to top5 for each sample_id
    for sample_id, scores in top_scores.items():
        print(f"Sample ID {sample_id}: {scores}")

# Run the function
output_dir = "clean_result"  # Ensure this matches the output directory used in filter.py
calculate_average_score(output_dir)