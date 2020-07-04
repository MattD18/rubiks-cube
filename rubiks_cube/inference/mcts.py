'''
Monte Carlo Tree Search Module

Following algorithm described in Sec 4.2 of
https://arxiv.org/pdf/1805.07470.pdf
'''
import copy

import tensorflow as tf
import numpy as np
from rubiks_cube.environment.cube import Cube
from rubiks_cube.agent.small_cnn import CNN

class Node():
    '''
    Node of search tree

    Attributes:
    -----------
    parent : Node
        parent node in search tree
    state : np.ndarray
        represents rubix cube state for this node
    action_taken_string : str
        string representing the action taken at this node's state at the
        current step of the tree search
    cube_moves : list
        string names of actions available to cube
    model : tf.keras.Model
    c : float
        exploration hyperparameter
    v : float
        virtual loss hyperparametr
    children : {str : Node}
        dict of children nodes in search tree
    N : dict {str : int}
        # of times each of the 12 actions has been taken from this state
    W : dict {str : float}
        the maximal value for taking each of the 12 actions from this state
    L : dict {str : float}
        the current "virtual loss" for each of the 12 actions from this state
    P : dict {str : float}
        the prior of taking each action from this state
    '''
    def __init__(self, state, model, c, v, parent=None):
        self.parent = parent
        self.state = copy.deepcopy(state)
        self.action_taken_string = None
        self.cube_moves = [action.__name__ for action in Cube().func_list]
        self.model = model
        self.c = c
        self.v = v
        self.children = self._init_children()
        self.N = self._init_N()
        self.W = self._init_W()
        self.L = self._init_L()
        self.P = self._init_P()

    def _init_N(self):
        '''
        Helper function to initialize N

        Returns:
        ---------
        N : dict {str : int}
            # of times each of the 12 actions has been taken from this state
        '''
        N = dict(zip(self.cube_moves, [0]*len(self.cube_moves)))
        return N

    def _init_W(self):
        '''
        Helper function to initialize W

        Returns:
        ---------
        W : dict {str : float}
            the maximal value for taking each of the 12 actions from this state
        '''
        #create local copy of node state
        node_state = copy.deepcopy(self.state)
        # convert to tensor
        node_state = tf.expand_dims(tf.convert_to_tensor(node_state), 0)
        # pass through Q function approximator (and normalize) to get probs
        action_values = self.model(node_state, training=False).numpy()[0]
        # create dict
        W = dict(zip(self.cube_moves, action_values))
        return W

    def _init_L(self):
        '''
        Helper function to initialize L

        Returns:
        ---------
        L : dict {str : float}
            the current "virtual loss" for each of the 12 actions from this state
        '''
        L = dict(zip(self.cube_moves, [0]*len(self.cube_moves)))
        return L

    def _init_P(self):
        '''
        Helper function to initialize P

        Returns:
        ---------
        P : dict {str : float}
            the prior of taking each action from this state

        '''
        #create local copy of node state
        node_state = copy.deepcopy(self.state)
        # convert to tensor
        node_state = tf.expand_dims(tf.convert_to_tensor(node_state), 0)
        # pass through Q function approximator (and normalize) to get probs
        action_probs = tf.keras.activations.softmax(self.model(node_state, training=False)).numpy()[0]
        # create dict
        P = dict(zip(self.cube_moves, action_probs))
        return P

    def _init_children(self):
        '''
        Helper function to initialize children

        Returns:
        ---------
        children : dict {str : Node}

        '''
        children = dict(zip(self.cube_moves, [None]*len(self.cube_moves)))
        return children

    def get_Q_st(self):
        '''
        Calculate MCTS Q value for each action in current state

        Returns:
        -----------
        Q_st : np.array
            MCTS Q(st, a) for each of 12 possible actions
        '''
        # convert W, L to numpy arrays
        W_st = np.array(list(self.W.values()))
        L_st = np.array(list(self.L.values()))
        # for each action, take difference of W(a) and L(a)
        Q_st = W_st - L_st
        return Q_st

    def get_U_st(self):
        '''
        Calculate U value for each action in current state

        Returns:
        -----------
        U_st : np.array
            U(st, a) for each of 12 possible actions
        '''
        # convert P, N to numpy arrays
        P_st = np.array(list(self.P.values()))
        N_st = np.array(list(self.N.values()))
        # follow equation in sec 4.2 of Solving the Rubik’s Cube Without Human Knowledge
        U_st = self.c*P_st*np.sqrt(N_st.sum()) / (1 + N_st)
        return U_st

    def update_memory(self, q_current_state):
        '''
        update node using q in backpropagation state of tree search
        '''
        if self.action_taken_string:
            self.W[self.action_taken_string] = max(self.W[self.action_taken_string], q_current_state)
            self.N[self.action_taken_string] += 1
            self.L[self.action_taken_string] += self.v
            self.action_taken_string = None

def mcts_solve(model, shuffled_cube, c=.1, v=.1, num_searches=100, verbose=False):
    '''
    Attempt to solve cube via Monte carlo tree search

    Parameters:
    -----------
    model : tf.keras.Model
        Q function approximator
    shuffled_cube : rubiks_cube.environment.cube.Cube()
        rubix cube object to be solved
    c : float
        exploration hyperparameter
    v : float
        virtual loss hyperparameter
    num_searches : int
        # of iterations to search for
    verbose : false
        print output on solving progress

    Returns:
    --------
    solved : boolean
    shuffled_cube : rubiks_cube.environment.cube.Cube()
    '''
    #initial condiations
    solved = False
    solved_cube = Cube()
    cube_state = copy.deepcopy(shuffled_cube.state)
    root = Node(cube_state, model, c, v, parent=None)
    # perform search
    for i in range(num_searches):
        # 1) Selection
        if verbose:
            print("Selection")
        #start search at inital state
        current_node = root
        # traverse search tree until leaf node encountered
        # Every simulation starts from the root node and iteratively selects actions by following a tree
        # policy until an unexpanded leaf node, sτ , is reached
        has_children = (sum(map(lambda x: x is None, current_node.children.values())) == 0)
        while has_children:
            #calculate values for current node
            Q_st = current_node.get_Q_st()
            U_st = current_node.get_U_st()
            # select "best" action to perform
            A_st = np.argmax(U_st + Q_st)
            A_st_string = current_node.cube_moves[A_st]
            if verbose:
                print(f"Enter Selection: {A_st_string}")
            #save action taken
            current_node.action_taken_string = A_st_string
            #move to next node
            current_node = current_node.children[A_st_string]
            has_children = (sum(map(lambda x: x is None, current_node.children.values())) == 0)

        #check if cube has been solved
        if (current_node.state == solved_cube.state).all():
            if verbose:
                print("Cube is solved")
            solved = True
            shuffled_cube.set_state(current_node.state)
            break

        # 2) Expansion
        if verbose:
            print("Expansion")
        # Once a leaf node, sτ , is reached, the state is expanded by adding the children of s
        for move in shuffled_cube.func_list:
            shuffled_cube.set_state(current_node.state)
            move()
            new_state = copy.deepcopy(shuffled_cube.state)
            new_node = Node(new_state, model, c, v, parent=current_node)
            #add resulting states to current node's children
            current_node.children[move.__name__] = new_node

        # 3) Simulation
        if verbose:
            print("Simulation")
        # make copy of current state for simulation
        current_state = copy.deepcopy(current_node.state)
        # convert current state to tensor
        current_state = tf.expand_dims(tf.convert_to_tensor(current_state), 0)
        # find max Q for in current state
        q_current_state = model(current_state, training=False).numpy()[0].max()


        # 4) Backpropagation
        if verbose:
            print("Backpropagation")
        # update nodes with results of simulation
        current_node.update_memory(q_current_state)
        # traverse tree
        while current_node.parent is not None:
            # update all past parents with q value from current state
            current_node = current_node.parent
            current_node.update_memory(q_current_state)

        if verbose:
            if i == num_searches:
                print("Time Out")
            else:
                print('--------------')

    return solved, shuffled_cube

if __name__ == "__main__":

    model = CNN()
    shuffled_cube = Cube()
    shuffled_cube.shuffle(2)
    solved, solved_cube = mcts_solve(model, shuffled_cube, c=.1, v=.1, num_searches=100, verbose=True)