import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
import pickle

def dominate(a, b):
    if a[0] < b[0] and a[1] < b[1]:
        return 1
    elif b[0] < a[0] and b[1] < a[1]:
        return -1
    else:
        return 0

def arkiv_proccess(history):
    arkiv = history[0][:]
    final_history = [arkiv[:]]  

    for i in range(1, len(history)):
        current_gen = history[i]
        valid_new_idx = []
        remove_arkv_idx = []

        for j in range(len(current_gen)):
            is_valid = True
            for k in range(len(arkiv)):
                check = dominate(arkiv[k], current_gen[j])
                if check == 1:
                    is_valid = False
                    break
                elif check == -1:
                    remove_arkv_idx.append(k)
            if is_valid:
                valid_new_idx.append(j)

        remove_arkv_idx = list(set(remove_arkv_idx))
        arkiv = [ind for idx, ind in enumerate(arkiv) if idx not in remove_arkv_idx]

        for j in valid_new_idx:
            arkiv.append(current_gen[j])

        final_history.append(arkiv[:])  
    return final_history

def visualize_process(final_history, objective_labels=["L_RSR", "L_GPR"], interval=500):
    num_generations = len(final_history)

    # Convert to numpy arrays if not already
    final_history = [np.array(gen) for gen in final_history]

    # Determine bounds across all generations
    min_obj1 = min(np.min(gen[:, 0]) for gen in final_history)
    max_obj1 = max(np.max(gen[:, 0]) for gen in final_history)
    min_obj2 = min(np.min(gen[:, 1]) for gen in final_history)
    max_obj2 = max(np.max(gen[:, 1]) for gen in final_history)

    padding_obj1 = (max_obj1 - min_obj1) * 0.1
    padding_obj2 = (max_obj2 - min_obj2) * 0.1
    xlim = (min_obj1 - padding_obj1, max_obj1 + padding_obj1)
    ylim = (min_obj2 - padding_obj2, max_obj2 + padding_obj2)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter([], [], c='red')
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)

    def update(frame):
        data = final_history[frame]
        scatter.set_offsets(data)
        ax.set_title(f"Pareto Front - Generation {frame + 1}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=num_generations, interval=interval, blit=True)
    plt.close(fig)
    return ani


class DataLoader:    
    def __init__(self, retri_dir=None, answer_dir=None):
        if retri_dir:
            self.retri_dir = retri_dir
        if answer_dir:
            self.answer_dir = answer_dir

    def take_retri_data(self, sample_id):
        sample_dir = os.path.join(self.retri_dir, str(sample_id))
        metadata_path = os.path.join(sample_dir, "metadata.json")
        retri_imgs_path = os.path.join(sample_dir, "retri_images.pkl")
        with open(metadata_path, "r") as f:
            data = json.load(f)
            
        question = data["question"]
        answer = data["answer"]
        query = data["keyword"]
        gt_basenames = data["gt_basenames"]
        retri_basenames = data["topk_basenames"]
        
        with open(retri_imgs_path, "rb") as f:
            retri_imgs = pickle.load(f)
        
        return question, answer, query, gt_basenames, retri_basenames, retri_imgs
    
    def take_answer_data(self, sample_id):
        sample_dir = os.path.join(self.answer_dir, str(sample_id))
        answers_path = os.path.join(sample_dir, "answers.json")
        with open(answers_path, "r") as f:
            data = json.load(f)
            
        golden_answers = data["predictions"]
    def __len__(self):
        return len(os.listdir(self.retri_dir))
    
    def take_data(self, sample_id):
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = self.take_retri_data(sample_id)
        golden_answers = self.take_answer_data(sample_id)
        return question, answer, query, golden_answers, gt_basenames, retri_basenames, retri_imgs
    


class ImagesLoader:
    def __init__(self, path, img_dir):
        self.img_dir = img_dir
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]    

    def take_data(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        image_items = self.data[idx]['images']
        paths = []
        gt_paths = []
        for path, label in image_items.items():
            img_path = os.path.join(self.img_dir, path)
            paths.append(img_path)
            if label == 1:
                gt_paths.append(img_path)
                
        return question, answer, paths, gt_paths
    
    def __len__(self):
        return len(self.data)
    
def greedy_selection(scores):
    valid_indices = np.where(scores[:, 0] < 1)[0]
    if len(valid_indices) > 0:
        success_retri = True
        best_idx = valid_indices[np.argmin(scores[:, 1][valid_indices])]
    else:
        success_retri = False
        best_idx = np.argmin(scores[:, 0])
    return scores[best_idx], success_retri