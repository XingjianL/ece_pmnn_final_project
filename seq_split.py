# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:45:23 2024

@author: somsh
"""

import pandas as pd
import os
from tqdm import tqdm
root='csv_data'

for j in range(1,10):
    Player_no=j
    Player = 'Player_'+str(Player_no)
    
    for k in range(0,8):
        file_no=k
        file_path=str(file_no)+".csv"
        file_idx_text=str(file_no)+"_new_sequences.txt"
        
        sequence_file_path = root+"/"+Player+"/"+file_idx_text
        csv_file_path=root+"/"+Player+"/"+file_path
        
        
        data = pd.read_csv(csv_file_path) 
        
        with open(sequence_file_path, 'r') as file:
            sequence_indices = [int(line.strip()) for line in file.readlines()]
        
        out_directory=f"./parsed_sequence_data/{Player}_{file_no}"
        
        # if not os.path.exists(out_directory):
        #     os.makedirs(out_directory)
        
        for i in tqdm(range(len(sequence_indices))):
            if i<len(sequence_indices)-1:
                split_data=data[sequence_indices[i]:sequence_indices[i+1]].copy()
            else: 
                split_data=data[sequence_indices[i]:].copy()

            initial_pos = split_data[['px', 'py', 'pz']].iloc[0]
            initial_time = split_data["time_incr"].iloc[0]
            split_data.loc[:,['px', 'py', 'pz']] = split_data.loc[:,['px', 'py', 'pz']].subtract(initial_pos)
            split_data.loc[:,"time_passed"]=split_data.loc[:,"time_incr"].cumsum().subtract(initial_time)
            split_data.reset_index(inplace=True)
            split_data=split_data.drop(["time_incr"],axis=1)
            split_data=split_data.drop(["index"],axis=1)
            if split_data.loc[:,"time_passed"].iloc[-1] < 4.9:
                print(out_directory, file_no, i, " not 5 seconds: ", split_data.loc[:,"time_passed"].iloc[-1])
                continue
            split_data.to_csv(f"{out_directory}_{str(i)}.csv",index=False)
    