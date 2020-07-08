import pytest

from rubiks_cube.inference.mcts import Node, mcts_solve
from rubiks_cube.agent.small_cnn import CNN
from rubiks_cube.environment.cube import Cube

def test_node_init():
    model = CNN()
    c = Cube()
    state = c.state
    node = Node(state, model, .4, .1)
    assert (node.cube_moves[2] == 'right') \
        & (len(node.P.keys()) == 12)

def test_mcts_solve_call():
    model = CNN()
    shuffled_cube = Cube()
    shuffled_cube.shuffle(2)
    solved, solved_cube = mcts_solve(model, shuffled_cube, c=.1, v=.1, num_searches=100, verbose=False)
    if solved:
        assert solved_cube == Cube()
    else: 
        assert solved_cube != Cube()
    