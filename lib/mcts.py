'''
module to perform Monte-Carlo Tree Search

'''

import copy

import numpy as np
import tensorflow as tf

from lib.cube import Cube



class Node():
    '''
    Node of search tree

    Attributes:
    -----------
    parent : Node
        parent node in search tree
    children : list
        list of children nodes in search tree
    state : np.ndarray
        represents a 3x3x3 rubix cube as a 3x3x3 numpy array where each piece
        has a unique integer id
    action_taken_string : str
        string representing the action taken at this node's state at the
        current step of the tree search
    N : dict
        counter for the number of times each action has been tried from current state
    W : dict
        TO DO
    L : dict
        TO DO
    P : dict
        TO DO
    '''
    def __init__(self, state, cube_moves, model, c, parent=None):
        self.parent = parent
        self.children = []
        self.state = copy.deepcopy(state)
        self.action_taken_string = None
        self.cube_moves = cube_moves
        self.model = model
        self.c = c
        self.N = self._init_N()
        self.W = self._init_W()
        self.L = self._init_L()
        self.P = self._init_P()

    def _init_N(self):
        N = dict(zip(self.cube_moves, [0]*len(self.cube_moves)))
        return N
    def _init_W(self):
        W = dict(zip(self.cube_moves, [0]*len(self.cube_moves)))
        return W

    def _init_L(self):
        L = dict(zip(self.cube_moves, [0]*len(self.cube_moves)))
        return L

    def _init_P(self):
        node_state = copy.deepcopy(self.state)
        node_state = tf.convert_to_tensor(node_state)
        node_state = tf.expand_dims(node_state, 0)
        action_probs = tf.keras.activations.softmax(self.model(node_state)).numpy()[0]
        P = dict(zip(self.cube_moves, action_probs))
        return P

    def get_Q_st(self):
        W_st = np.array(list(self.W.values()))
        L_st = np.array(list(self.L.values()))
        Q_st = W_st - L_st
        return Q_st

    def get_U_st(self):
        P_st = np.array(list(self.P.values()))
        N_st = np.array(list(self.N.values()))
        U_st = self.c*P_st*np.sqrt(N_st.sum()) / (1 + N_st)
        return U_st

    def update_memory(self, q_current_state):
        self.W[self.action_taken_string] = max(self.W[self.action_taken_string], q_current_state)
        self.N[self.action_taken_string] += 1
        self.action_taken_string = None

class MCTS():
    '''
    class to perform mcts on a given cube

    Attributes:
    ----------
    model : tensorflow.keras.Model
        Q function approximator
    c : float
        exploration hyper parameter
    v : float
        virtual loss hyper parameter
    num_searches : int
        number of searches to perform
    solved_cube : lib.Cube
        solved cube used as reference to check for solved state
    cube_moves : list
        list of valid cube moves
    '''
    def __init__(self, model, c=.1, v=.5, num_searches=50):
        self.model = model
        self.c = c
        self.v = v
        self.num_searches = num_searches
        self.solved_cube = Cube()
        self.cube_moves = []
        for move in self.solved_cube.func_list:
            self.cube_moves.append(str(move).split()[2].split('.')[1])

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

    def solve(self, cube, verbose=False, c=None, v=None, num_searches=None):
        '''
        attempts to solve cube using MCTS

        Parameters:
        -----------
        cube : lib.Cube
        '''
        if c:
            self.c = c
        if v:
            self.v = v
        if num_searches:
            self.num_searches = num_searches


        solved = False
        root = Node(cube.state, self.cube_moves, self.model, self.c)
        for i in range(self.num_searches):

            #selection
            if verbose:
                print("Selection")
            current_node = root
            while current_node.children: #while not leaf node
                Q_st = current_node.get_Q_st()
                U_st = current_node.get_U_st()
                A_st = np.argmax(U_st + Q_st)
                A_st_string = self.cube_moves[A_st]
                if verbose:
                    print(f"Enter Selection: {A_st_string}")
                #update loss
                current_node.L[A_st_string] += self.v
                #save action taken
                current_node.action_taken_string = A_st_string
                #move to next node
                current_node = current_node.children[A_st]

            #check for termination
            if (current_node.state == self.solved_cube.state).all():
                if verbose:
                    print("Cube is solved")
                solved = True
                cube.set_state(current_node.state)
                break

            #expansion
            if verbose:
                print("Expansion")
            for move in cube.func_list:
                cube.set_state(current_node.state)
                move()
                new_state = Node(cube.state,
                                 self.cube_moves,
                                 self.model,
                                 self.c,
                                 parent=current_node)
                current_node.children.append(new_state)

            #simulation
            if verbose:
                print("Simulation")

            current_state = copy.deepcopy(current_node.state)
            current_state = tf.convert_to_tensor(current_state)
            current_state = tf.expand_dims(current_state, 0)

            a_current_state = self.model(current_state).numpy()[0].argmax()
            current_node.action_taken_string = self.cube_moves[a_current_state]

            q_current_state = self.model(current_state).numpy()[0].max() #are q/v the same?


            #backpropagation
            if verbose:
                print("Backpropagation")
            current_node.update_memory(q_current_state)
            while current_node.parent is not None:
                if verbose:
                    print("Propagating")
                current_node = current_node.parent
                current_node.update_memory(q_current_state)

            if verbose:
                if i == self.num_searches:
                    print("Time Out")
                else:
                    print('--------------')

        return solved, cube
