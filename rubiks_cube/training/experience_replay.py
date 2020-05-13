''' Experience replay algorithm as described in
Algorithm 1 of https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
'''
import numpy as np
import tensorflow as tf

from rubiks_cube.environment.cube import Cube
from rubiks_cube.agent.replay_buffer import ReplayBuffer

def train_via_experience_replay(model, loss_object, optimizer, 
                                num_epochs=10, 
                                buffer_size=128,
                                **episode_kwargs):
    '''
    Wrapper function to train model

    Parameters:
    -----------
    model : tf.keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    num_epochs : int
    buffer_size : int
    **episode_kwargs : keyword arguments to play_episde
    '''
    print("Training model")
    rb = ReplayBuffer(buffer_size=buffer_size)
    for i in range(num_epochs):
        play_episode(model, loss_object, optimizer, rb, **episode_kwargs)


def play_episode(model, loss_object, optimizer, replay_buffer, 
                    num_shuffles=5, 
                    max_time_steps=10, 
                    exploration_rate=.1, 
                    end_state_reward=1.0, 
                    batch_size=16, 
                    discount_factor=.9,
                    training=True):
    '''
    In a single episode, agent has max_time_steps moves to try to solve a cube randomly shuffled num_shuffle times

    Parameters:
    ------------
    model : tf.keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    replay_buffer : rubiks_cube.agent.replay_buffer.ReplayBuffer
    num_shuffles : int (>= 0)
    max_time_steps : int (>= 1)
    exploration_rate : float [0, 1]
    end_state_reward: float
    batch_size : int (>= 1)
    discount_factor: float
    training : boolean
    '''
    #Initialize cube state
    cube = Cube()
    solved_cube = Cube()
    s0 = cube.shuffle(num_shuffles)
    # convert cube state into tensor to feed into model
    st = tf.expand_dims(tf.convert_to_tensor(s0), 0) # (1, 3, 3, 3)
    #Play episode until solved or max_time_steps is reached
    for t in range(max_time_steps):
        #with some probability select a random action a_t
        if np.random.rand() < exploration_rate:
            at_index = np.random.randint(0, 12) #WARNING: Number of possible otations
        #otherwise select a_t = max_a Q(s_t,a)
        else:
            at_index = tf.math.argmax(model(st), 1).numpy()[0]
        # Execute action a_t and observe state s_t+1 and reward r_t
        at = cube.func_list[at_index]
        st1 = at()
        if cube == solved_cube:
            rt = end_state_reward
        else:
            rt = 0.
        # Store transition in replay buffer, convert state to numpy for convenience
        st_numpy = st.numpy()[0] # (3, 3, 3)
        transition = (st_numpy, at_index, rt, st1) # (np.array, int, float, np.array)
        replay_buffer.add(transition)
        #if training is enabled, update q function
        if training:
            update_q_function(model, loss_object, optimizer, replay_buffer, end_state_reward, batch_size, discount_factor)
        #if reward state has been reached, stop episode early
        if (rt == end_state_reward):
            break
        # convert next cube state into tensor to feed into model
        st = tf.expand_dims(tf.convert_to_tensor(st1), 0) # (1, 3, 3, 3)
    return cube

def update_q_function(model, loss_object, optimizer, replay_buffer, end_state_reward, batch_size=16, discount_factor=.9):
    '''
    update weights of model with a minibatch samples from replay_buffer

    model : tf.Keras.Model
    loss_object : tf.keras.losses
    optimizer : tf.keras.optimizer
    replay_buffer : rubiks_cube.agent.replay_buffer.ReplayBuffer
    end_state_reward: float
    batch_size : int (>= 1)
    discount_factor: float
    '''
    if replay_buffer.is_full():
        #grab batch values from replay buffer
        minibatch = replay_buffer.get_minibatch(batch_size)
        states, actions, rewards, next_states = unpack_minibatch(minibatch)
        #perform update
        with tf.GradientTape() as tape:
            rows = model(states, training=True)
            selector = tf.expand_dims(actions, 1)
            idx = tf.stack([tf.reshape(tf.range(rows.shape[0]), (-1, 1)), selector], axis=-1)
            predicted_q_vals = tf.gather_nd(rows, idx) # (bs, 1)
            y_vals = get_y_vals(rewards, next_states, model, discount_factor, end_state_reward) #(bs, 1)
            loss = loss_object(y_vals, predicted_q_vals)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))



def get_y_vals(rewards, next_states, model, discount_factor, end_state_reward):
        '''
        gets approximate reward values based on past experience
        y_j = r_j + I[st1 is not terminal] * gamma * max_a Q(st1, a)
        as in https://arxiv.org/pdf/1312.5602v1.pdf algorithm 1

        Parameters:
        ------------

        rewards : tf.Tensor (bs, )
        next_states : tf.Tensor (bs, 3, 3, 3)
        model : tf.keras.Model
        discount_factor : float
            discount applied to future values in Q function
        end_state_reward : float
        '''
        # convert tensor to np.array for easier processing
        rewards_numpy = rewards.numpy()
        next_state_q_numpy = model(next_states).numpy()
        # add additional term to non_terminal transitions
        non_terminal_mask = (rewards_numpy != end_state_reward)
        rewards_numpy[non_terminal_mask] += discount_factor * np.max(next_state_q_numpy[non_terminal_mask], 1)
        # return value as tensor
        y_vals = tf.expand_dims(tf.constant(rewards_numpy), 1)
        return y_vals




def unpack_minibatch(minibatch):
    '''
    unpacks minibatch into tensors for gradient update

    Parameters:
    mb : list
        list of transition tuples
    '''
    #gets current states, actions, and rewards from batch as tensors
    states = tf.stack([transition[0] for transition in minibatch], 0) # (bs, 3, 3, 3)
    actions = tf.constant([transition[1] for transition in minibatch]) # (bs, )
    rewards = tf.constant([transition[2] for transition in minibatch]) # (bs, )
    next_states = tf.stack([transition[3] for transition in minibatch], 0) # (bs, 3, 3, 3)
    return states, actions, rewards, next_states

