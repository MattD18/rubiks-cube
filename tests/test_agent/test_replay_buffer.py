import pytest

from rubiks_cube.agent import replay_buffer

def test_replaybuffer_empty_on_init():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    assert rb.buffer == [None] * 128

def test_replaybuffer_add():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    sample_transition = (0, 0, 0, 0)
    for i in range(5):
        rb.add(sample_transition)
    assert rb.buffer[4] == (0, 0, 0, 0)

def test_replaybuffer_add_wrap():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    sample_transition_one = (0, 0, 0, 0)
    for i in range(128):
        rb.add(sample_transition_one)
    sample_transition_two = (1, 1, 1, 1)
    for i in range(5):
        rb.add(sample_transition_two)
    assert (rb.buffer[4] == (1, 1, 1, 1)) & (rb.buffer[5] == (0, 0, 0, 0))

def test_replaybuffer_isfull_empty():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    sample_transition = (0, 0, 0, 0)
    for i in range(5):
        rb.add(sample_transition)
    assert not rb.is_full()

def test_replaybuffer_isfull_full():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    sample_transition_one = (0, 0, 0, 0)
    for i in range(128):
        rb.add(sample_transition_one)
    sample_transition_two = (1, 1, 1, 1)
    for i in range(5):
        rb.add(sample_transition_two)
    assert rb.is_full()

def test_replaybuffer_get_minibatch_batch_size():
    rb = replay_buffer.ReplayBuffer(buffer_size=128)
    sample_transition_one = (0, 0, 0, 0)
    for i in range(128):
        rb.add(sample_transition_one)
    sample_transition_two = (1, 1, 1, 1)
    for i in range(5):
        rb.add(sample_transition_two)
    batch = rb.get_minibatch(16)
    assert len(batch) == 16