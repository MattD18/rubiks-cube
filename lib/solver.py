'''
Module for rubix cube solver

'''
import datetime
import random
import copy

import numpy as np
import tensorflow as tf

from lib.cube import Cube


class CubeSolver():
    '''
    Main interface to solve rubix cube and train solver

    Attributes:
    -----------
    model : tensorflow.keras.Model
        Q function approximator
    current_time : str
        time of solver initialization, used for logging
    train_log_dir : str
        location of logging directory for tensorboard
    model_dir : str
        location of directory to save trained model weights
    '''

    def __init__(self, train_log_dir='../logs/gradient_tape/', model_dir='../models/'):
        self.model = None
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = train_log_dir
        self.model_dir = model_dir

    def train(self,
              exploration_rate_func,
              num_shuffles=5,
              num_episodes=5000,
              max_time_steps=10,
              epsilon=.99,
              decay_constant=10,
              end_state_reward=100,
              replay_buffer_capacity=1024,
              learning_rate=.005,
              clipnorm=5,
              batch_size=64,
              discount_factor=.95,
              validation_count=100,
              train_log_name=None,
              stop_on_solve=False):
        '''
        Trains Q function approximator

        Parameters
        ----------
        exploration_rate_func : function
            method to adjust exploration rate for e-greedy training
        num_shuffles : int
            number of times a cube will be shuffled in training episodes
        num_episodes : int
            number of episodes to train cube for (includes episodes used to fill
            replay buffer)
        max_time_steps : int
            number of time steps agent act on cube for in a given episode
        epsilon : float [0,1)
            base exploration rate for training algorithmm
        decay_constant : int
            constant used to scale exploration rate over training episodes
        end_state_reward : int
            reward given for moving cube into solved state
        replay_buffer_capcity : int
            size of experience replay buffer
        learning_rate : float
            learning rate for gradient descent algorithm
        clipnorm : float
            threshold at which to clip gradients
        batch_size : int
            size of minibatch for gradient updates
        discount_factor : float
            discount applied to future values in Q function
        validation_count : int
            number of cubes to use for validating during training
        train_log_name : str
            optional name for training log, defaults to timestamp
        stop_on_solve : bool
            training episodes will terminate if cube is solved
        '''
        assert self.model is not None, "model needs to be instantiated"
        assert issubclass(self.model.__class__, tf.keras.Model), "model must be subclass of tf.keras.Model"

        #gradient descent variables
        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             clipnorm=clipnorm)

        #train logging variables
        if train_log_name is None:
            train_log_dir = self.train_log_dir + self.current_time + '/train'
        else:
            train_log_dir = self.train_log_dir + train_log_name + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        validation_cubes = []
        for i in range(validation_count):
            val_cube = Cube()
            val_cube.shuffle(num_shuffles)
            validation_cubes.append(val_cube)

        #Rubix cube variables
        solved_cube = Cube()
        num_actions = len(solved_cube.func_list)
        #initialize replay memory
        replay_buffer = [None for i in range(replay_buffer_capacity)]
        replay_counter = -1

        for episode in range(num_episodes):
            #set exploration rate for episode
            exploration_rate = exploration_rate_func(epsilon, decay_constant, episode)
            #Initialize sequence s_1
            episode_cube = Cube()
            s1 = episode_cube.shuffle(num_shuffles)
            st = tf.convert_to_tensor(s1)
            st = tf.expand_dims(st, 0)
            for time_step in range(max_time_steps):
                #with some probability select a random action a_t
                if np.random.rand() < exploration_rate:
                    at_index = np.random.randint(0, num_actions)
                #otherwise select a_t= max_a Q(s_t,a)
                else:
                    at_index = tf.math.argmax(self.model(st), 1).numpy()[0]
                at = episode_cube.func_list[at_index]
                # Execute action a_t in emulator and observe image s_t+1 and reward r_t
                st1 = at()
                if episode_cube == solved_cube:
                    rt = end_state_reward
                else:
                    rt = 0
                #Store transition(st,at,rt,st+1) in replay buffer
                if replay_counter < replay_buffer_capacity - 1:
                    replay_counter += 1
                else:
                    replay_counter = 0
                replay_buffer[replay_counter] = (st, at_index, rt, st1)
                #when buffer is full sample random minibatch of replay buffer
                #to perform a gradient update step on the Q function approximator
                num_experiences = sum([0 if x is None else 1 for x in replay_buffer])
                if num_experiences == replay_buffer_capacity:
                    #grab batch values from replay buffer
                    batch = random.sample(replay_buffer, batch_size)
                    states = [element[0] for element in batch]
                    actions = [element[1] for element in batch]
                    rewards = [element[2] for element in batch]
                    actions = tf.convert_to_tensor(actions)
                    rewards = tf.convert_to_tensor(rewards)
                    #perform update
                    with tf.GradientTape() as tape:
                        rows = self.model(tf.squeeze(tf.stack(states, 1), 0))
                        selector = tf.expand_dims(actions, 1)
                        idx = tf.stack([tf.reshape(tf.range(rows.shape[0]), (-1, 1)), selector], axis=-1)
                        predicted_q_vals = tf.gather_nd(rows, idx)
                        reinforce_q_vals = self.get_reinforce_q_vals(batch, discount_factor)
                        reinforce_q_vals = tf.expand_dims(tf.stack(reinforce_q_vals), 1)
                        loss = loss_object(predicted_q_vals, reinforce_q_vals)

                    grads = tape.gradient(loss, self.model.trainable_variables)
                    try:
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    except ValueError:
                        import pdb;pdb.set_trace()
                    train_loss(loss)
                st = tf.convert_to_tensor(st1)
                st = tf.expand_dims(st, 0)
                if (rt == end_state_reward) and stop_on_solve:
                    break

            #evey episode, save training loss and reset loss counter
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=episode)
            train_loss.reset_states()

            #every 25 episodes, save val accuracy and average max q val, which
            #is an indicator of training progress as described in
            #https://arxiv.org/pdf/1312.5602v1.pdf section 5.1
            if episode % 25 == 0:

                avg_max_q = np.mean([self.model( \
                                     tf.expand_dims(tf.convert_to_tensor(val_cube.state), 0)) \
                                     .numpy() \
                                     .max() for val_cube in validation_cubes])
                with train_summary_writer.as_default():
                    tf.summary.scalar('avg_max_q', avg_max_q, step=episode)

                solve_count = 0
                for val_trial in range(validation_count):
                    val_trial_cube = Cube()
                    val_trial_cube.shuffle(num_shuffles)
                    solve_count += self.solve(val_trial_cube, max_time_steps)[0]
                val_acc = float(solve_count) / float(validation_count)
                with train_summary_writer.as_default():
                    tf.summary.scalar('val_acc', val_acc, step=episode)

    def get_reinforce_q_vals(self, batch, discount_factor):
        '''
        gets approximate reward values based on past experience
        y_j in https://arxiv.org/pdf/1312.5602v1.pdf algorithm 1

        Parameters:
        ------------

        batch : list
            minibatch of transitions stored in replay buffer
        discount_factor : float
            discount applied to future values in Q function
        '''
        reinforce_rewards = []
        for batch_element in batch:
            solved_state = Cube().state
            future_state = batch_element[3]
            reinforce_reward = batch_element[2]
            if not(future_state == solved_state).all():
                future_state_tensor = tf.convert_to_tensor(future_state)
                future_state_tensor = tf.expand_dims(future_state_tensor, 0)
                future_state_q = tf.math.reduce_max(self.model(future_state_tensor))
                reinforce_reward += discount_factor * future_state_q
            reinforce_rewards.append(reinforce_reward)
        return reinforce_rewards

    def solve(self, cube, max_time_steps, verbose=False):
        '''
        solves cube with current q function approximator

        Parameters:
        -----------
        cube : Cube
            rubix cube object to be solved
        max_time_steps : int
            maximum number of time steps allowed to solve cube
        verbose : boolean
            whether to print steps taken to solve
        '''
        solved_cube = Cube()
        solved = False
        solver_steps = []
        instance_s1 = copy.deepcopy(cube.state)
        instance_st = tf.convert_to_tensor(instance_s1)
        instance_st = tf.expand_dims(instance_st, 0)
        for time_step in range(max_time_steps):
            instance_at_index = tf.math.argmax(self.model(instance_st), 1).numpy()[0]
            instance_at = cube.func_list[instance_at_index]
            solver_steps.append(instance_at.__name__)
            if verbose:
                print(instance_at)
            instance_st1 = instance_at()
            if cube == solved_cube:
                solved = True
                break
            instance_st = tf.convert_to_tensor(instance_st1)
            instance_st = tf.expand_dims(instance_st, 0)
        return solved, cube, solver_steps

    def load_model_weights(self, weights_dir):
        '''
        Load model weights from weights_dir
        see demo: https://colab.research.google.com/drive/172D4jishSgE3N7AO6U2OKAA_0wNnrMOq#scrollTo=OOSGiSkHTERy
        '''
        assert issubclass(self.model.__class__, tf.keras.Model), "model must be subclass of tf.keras.Model"
        #model must be called on data once
        sample_state = np.arange(0, 27).reshape(3, 3, 3)
        sample_state = tf.convert_to_tensor(sample_state)
        sample_state = tf.expand_dims(sample_state, 0)
        self.model(sample_state)
        self.model.load_weights(weights_dir)

    def save_model_weights(self, weights_dir_name=None):
        '''
        Saves weights of current model

        weights_dir_name : str
            optional name for weights directory, defaults to timestamp
        '''
        if weights_dir_name is None:
            weights_dir = self.model_dir + self.current_time + '/weights'
        else:
            weights_dir = self.model_dir + weights_dir_name + '/weights'
        self.model.save_weights(weights_dir, save_format='tf')
