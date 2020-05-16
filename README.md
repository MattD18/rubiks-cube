## Welcome

This repo documents my work to train an agent to solve a Rubik's cube using a variant of deep reinforcement learning inspired by Playing Atari with Deep Reinforcement Learning (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).


### Part 1: Problem Set-Up and Deep Q-Learning ([link](notebooks/Part%201.ipynb))

### Part 2: Adding Monte-Carlo Tree Search and GPU Training via Amazon EC2 ([link](notebooks/Part%202.ipynb))



Project Structure:

-RubiksCube
    -main_train.py
        -include validation method
    -training
        -experience_replay.py
    -environment
        -cube.py
    -q_function
        -small_cnn.py
    -inference
        -mcts.py
        -greedy.py
-notebooks
-saved_weights
-training_log
    -gradient_tape
-train.sh

TO DO List:

Set up training main + logging