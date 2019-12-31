import sys
import datetime
import time
import csv
sys.path.append('..')


import numpy as np


from lib.cube import Cube
from lib.mcts import MCTS
from lib.solver import CubeSolver
from lib.models import WideNet

config = {}

config['model_params'] = {'regularization_constant': 0.05}



model = WideNet(**config['model_params'])
solver = MCTS(model)

model_path = '../models/base_model_v2_20191224023143/weights'

solver.load_model_weights(model_path)
n_trials = 100

if __name__ == "__main__":
    
    for n_shuffles in [10]:
        for ns in [10]:
            for c in [.1]:
                for v in [.1]:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    num_solved = 0
                    solved_times = []
                    not_solved_times = []
                    for trial in range(n_trials):
                        cube = Cube()
                        cube.shuffle(num_shuffles=n_shuffles,verbose=False)
                        trial_start = time.time()
                        solved,solved_cube = solver.solve(cube,c=c,v=v,num_searches=ns,verbose=False)
                        trial_end = time.time()
                        if solved:
                            solved_times.append(trial_end-trial_start)
                            num_solved +=1 
                        else:
                            not_solved_times.append(trial_end-trial_start)
                    solve_rate = num_solved/n_trials
                    avg_solve_time = np.mean(solved_times)    
                    avg_not_solve_time = np.mean(not_solved_times)   
                    avg_overall_time = np.mean(solved_times + not_solved_times)
                    results = {'model_path':model_path, 
                                    'timestamp':timestamp,
                                    'num_shuffles':n_shuffles,
                                    'n_trials':n_trials,
                                    'c':c,
                                    'v':v,
                                    'n_searches':ns,
                                    'avg_solve_time':avg_solve_time,
                                    'avg_not_solve_time':avg_not_solve_time,
                                    'avg_overall_time':avg_overall_time,
                                    'solve_rate':solve_rate}
    
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = f'../data/mcts_results_{now}.csv'


    with open(save_path, 'w') as f:
        for key in results.keys():
            f.write("%s,%s\n"%(key,results[key]))

   
