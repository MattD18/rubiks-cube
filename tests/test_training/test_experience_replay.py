import pytest

import numpy as np
import random
import tensorflow as tf

from rubiks_cube.agent.small_cnn import CNN
from rubiks_cube.agent.replay_buffer import ReplayBuffer
from rubiks_cube.training.experience_replay import play_episode, unpack_minibatch, get_y_vals, update_q_function
from rubiks_cube.training.experience_replay import train_via_experience_replay
from rubiks_cube.environment.cube import Cube

from rubiks_cube.agent import replay_buffer


def test_experience_replay_play_episode():
    np.random.seed(5)
    random.seed(5)
    tf.random.set_seed(5)
    model = CNN()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    rb = replay_buffer.ReplayBuffer()
    cube = play_episode(model, loss_object, optimizer, rb, training=False)
    assert (cube.state == np.array([[[26, 15,  0],
                                    [ 3,  4, 11],
                                    [ 8, 25, 18]],

                                    [[ 9, 10, 19],
                                    [12, 13, 14],
                                    [ 5, 16, 17]],

                                    [[24, 21,  6],
                                    [ 1, 22,  7],
                                    [ 2, 23, 20]]])).all()

def test_unpack_minibatch_shapes():
    batch_size = 16
    c = Cube()
    minibatch = [(c.state, 1, 1, c.state)] * batch_size
    states, actions, rewards, next_states = unpack_minibatch(minibatch)
    assert (states.shape == tf.TensorShape([batch_size, 3, 3, 3])) \
         & (actions.shape == tf.TensorShape([batch_size])) \
         & (rewards.shape == tf.TensorShape([batch_size])) \
         & (next_states.shape == tf.TensorShape([batch_size, 3, 3, 3]))

def test_get_y_vals():
    c = Cube()
    end_state_reward = 1.
    discount_factor = .9
    rewards = tf.constant([0, end_state_reward, end_state_reward, 0])
    next_states = tf.stack([c.state]*4,0)
    model = CNN()
    y_vals = get_y_vals(rewards, next_states, model, discount_factor, end_state_reward)
    assert (y_vals.numpy()[[1,2]].flatten() == np.array([end_state_reward, end_state_reward])).all() \
        & (y_vals.numpy()[0].flatten() > 0) & (y_vals.numpy()[3].flatten() > 0)


def test_update_q_function():
    model = CNN()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    end_state_reward = 1.
    discount_factor = 9
    batch_size = 4
    rb = ReplayBuffer(buffer_size=batch_size)
    c = Cube()
    st = c.state
    at = 3
    rt = 0.
    st1 = st
    transition = (st, at, rt, st1)
    for i in range(batch_size):
        rb.add(transition)
    model.build(input_shape=tf.TensorShape([batch_size, 3, 3, 3]))
    before_update_vars = [tf.identity(var) for var in model.trainable_variables]
    update_q_function(model, loss_object, optimizer, rb, end_state_reward, batch_size, discount_factor)
    after_update_vars = model.trainable_variables

    for (b,a) in zip(before_update_vars, after_update_vars):
        assert (a != b).numpy().any()


def test_train_via_experience_replay_change_weights():
    model = CNN()
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    model.build(input_shape=tf.TensorShape([4, 3, 3, 3]))
    before_update_vars = [tf.identity(var) for var in model.trainable_variables]
    train_via_experience_replay(model, loss_object, optimizer,
                                num_epochs=5, buffer_size=4,
                                num_shuffles=1, 
                                max_time_steps=5, 
                                exploration_rate=.1, 
                                end_state_reward=1.0, 
                                batch_size=4, 
                                discount_factor=.9,
                                training=True)
    after_update_vars = model.trainable_variables
    for (b,a) in zip(before_update_vars, after_update_vars):
        assert (a != b).numpy().any()
