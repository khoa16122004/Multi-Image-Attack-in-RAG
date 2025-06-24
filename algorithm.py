import torch
import random
from copy import deepcopy
import numpy as np
import os
import pickle
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm
from config import Config
import json

class NSGAII:
    def __init__(self, 
                 population_size, 
                 mutation_rate, 
                 F, 
                 w, 
                 h,
                 max_iter, 
                 fitness, # multi score
                 std,
                 sample_id,
                 log_dir,
                 n_k,
                 ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.F = F
        self.w = w
        self.h = h
        self.max_iter = max_iter
        self.fitness = fitness  # function
        self.std = std
        self.nds = NonDominatedSorting()
        self.sample_id = sample_id
        self.log_dir = os.path.join(log_dir, f"{self.fitness.retriever_name}_{self.fitness.reader_name}_{self.std}", str(self.sample_id))
        self.n_k = n_k
        os.makedirs(self.log_dir, exist_ok=True)

        

    def gaussian_patch_mutation(self, P, std=0.1, patch_size=16):
        P_ = deepcopy(P)
        _, _, H, W = P.shape
        x = torch.randint(0, W - patch_size, (1,))
        y = torch.randint(0, H - patch_size, (1,))
        noise = torch.randn_like(P[:, :, y:y+patch_size, x:x+patch_size]) * std
        P_[:, :, y:y+patch_size, x:x+patch_size] += noise
        return P_

    def calculating_crowding_distance(self, F):
        infinity = 1e+14

        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)
        else:

            # sort each column and get index
            I = np.argsort(F, axis=0, kind='mergesort')

            # now really sort the whole array
            F = F[I, np.arange(n_obj)]

            # get the distance to the last element in sorted list and replace zeros with actual values
            dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

            index_dist_is_zero = np.where(dist == 0)

            dist_to_last = np.copy(dist)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]

            dist_to_next = np.copy(dist)
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

            # normalize all the distances
            norm = np.max(F, axis=0) - np.min(F, axis=0)
            norm[norm == 0] = np.nan
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            # if we divided by zero because all values in one columns are equal replace by none
            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            # sum up the distance to next and last and norm by objectives - also reorder from sorted list
            J = np.argsort(I, axis=0)
            crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # replace infinity with a large number
        crowding[np.isinf(crowding)] = infinity
        return crowding


    def solve(self):
        P = torch.rand(self.population_size, 3, self.w, self.h).cuda() * self.std
        P_retri_score, P_reader_score, P_adv_imgs = self.fitness(P)
        
        self.history = []
        self.img_history = []
        for iter in tqdm(range(self.max_iter)):
            r1, r2, r3 = [], [], []
            for i in range(self.population_size):
                choices = [idx for idx in range(self.population_size) if idx != i]
                selected = random.sample(choices, 3)
                r1.append(selected[0])
                r2.append(selected[1])
                r3.append(selected[2])

            r1 = torch.tensor(r1, dtype=torch.long, device="cuda")
            r2 = torch.tensor(r2, dtype=torch.long, device="cuda")
            r3 = torch.tensor(r3, dtype=torch.long, device="cuda")
            
            x1 = deepcopy(P[r1])
            x2 = deepcopy(P[r2])
            x3 = deepcopy(P[r3])

            # print(x1.shape, x2.shape, x3.shape)
            
            v = x1 + self.F * (x2 - x3)
            O = torch.clamp(v, -self.std, self.std)


            O_retri_score, O_reader_score, O_adv_imgs = self.fitness(O)
            
            # pool
            pool = torch.cat([P, O], dim=0)  # (population_size, 2, ...)
            pool_retri_score = np.concatenate([P_retri_score, O_retri_score], axis=0)  # (population_size, 2)
            pool_reader_score = np.concatenate([P_reader_score, O_reader_score], axis=0)  # (population_size, 2)
            pool_fitness = np.column_stack((pool_retri_score, pool_reader_score))  # (population_size, 2)
            pool_adv_imgs = P_adv_imgs + O_adv_imgs
         
         
            # NSGA-II selection
            selected_indices, fronts = self.NSGA_selection(pool_fitness)
            P_retri_score = pool_retri_score[selected_indices]
            P_reader_score = pool_reader_score[selected_indices]
            P = pool[selected_indices]
            P_adv_imgs = [pool_adv_imgs[i] for i in selected_indices]
            
            
            
            rank_0_indices = fronts[0]  # Get indices of the first Pareto front
            rank_0_individuals = pool[rank_0_indices]
            rank_0_retri_scores = pool_retri_score[rank_0_indices]
            rank_0_reader_scores = pool_reader_score[rank_0_indices]  
            rank_0_adv_imgs = [pool_adv_imgs[i] for i in rank_0_indices]
           
            self.history.append(np.column_stack([rank_0_retri_scores, rank_0_reader_scores]))
            self.img_history.append(rank_0_adv_imgs)
            self.best_individual = rank_0_individuals
            self.best_retri_score = rank_0_retri_scores
            self.best_reader_score = rank_0_reader_scores 
            self.rank_0_adv_imgs = rank_0_adv_imgs
            
        
        self.save_logs()
            
    def save_logs(self):
        score_log_file = os.path.join(self.log_dir, f"scores_{self.n_k}.pkl") 
        invidual_log_file = os.path.join(self.log_dir, f"individuals_{self.n_k}.pkl")
        img_pkl_file = os.path.join(self.log_dir, f"images_{self.n_k}.pkl")
        adv_img_file = os.path.join(self.log_dir, f"adv_{self.n_k}.pkl")
        adv_history_file = os.path.join(self.log_dir, f"adv_history_{self.n_k}.pkl")
        answer_file = os.path.join(self.log_dir, f"answers_{self.n_k}.json")
        
        # inference
        final_selection_adv_img, retri_success = self.final_selection()
        adv_answer = self.fitness.reader.image_to_text(
            qs=self.fitness.question,
            img_files=self.fitness.top_adv_imgs + [final_selection_adv_img]
        )[0]
        answers = {
            "golden_answer": self.fitness.answer,
            "adv_answer": adv_answer,
            "retri_success": retri_success
        }
        
        with open(answer_file, "w") as f:
            json.dump(answers, f, indent=4)
        
        with open(adv_img_file, 'wb') as f:
            pickle.dump(final_selection_adv_img, f)

        with open(score_log_file, 'wb') as f:
            pickle.dump(self.history, f)
        with open(invidual_log_file, 'wb') as f:
            pickle.dump(self.best_individual, f)

        # Save list of images directly
        with open(img_pkl_file, 'wb') as f:
            pickle.dump(self.rank_0_adv_imgs, f)
            
        with open(adv_history_file, 'wb') as f:
            pickle.dump(self.img_history, f)
    
    def final_selection(self):
        
        valid_indices = np.where(self.best_retri_score < 1)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[np.argmin(self.best_reader_score[valid_indices])]
            success_full = True
        else:
            success_full = False
            best_idx = np.argmin(self.best_retri_score)
        return self.rank_0_adv_imgs[best_idx], success_full

        
    
    def NSGA_selection(self, pool_fitness):
        
        fronts = self.nds.do(pool_fitness, n_stop_if_ranked=self.population_size) # front ranked
        survivors = []
        for k, front in enumerate(fronts):
            crowding_of_front = self.calculating_crowding_distance(pool_fitness[front])
            sorted_indices = np.argsort(-crowding_of_front)
            front_sorted = [front[i] for i in sorted_indices]
            for idx in front_sorted:
                if len(survivors) < self.population_size:
                    survivors.append(idx)
                else:
                    break
            if len(survivors) >= self.population_size:
                break
        return survivors, fronts
    
     
    
