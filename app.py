from flask import Flask, render_template, url_for, send_file, request, g
from lib.cube import Cube
from lib.solver import CubeSolver
from lib.models import CNN

import json

#TO DO
#add a cube reset button, user warning that refreshing page won't reset cube
#add a solve button
#work out shuffle dynamics
#possibly allow for interchanging of models make in a command line argument

app = Flask(__name__)

# def main():
#     session_name = '3_shuffle_large_run_10000_50000'
#     solver = CubeSolver()
#     solver.model = CNN(embed_dim=100,
#                        num_filters=50,
#                        kernel_size=(2, 2, 2),
#                        regularization_constant=.1)
#     solver.load_model_weights('models/3_shuffle_large_run/weights')

with app.app_context():
    CUBE = Cube() #check if this is right way
    SOLVER = CubeSolver()
    SOLVER.model = CNN(embed_dim=100,
                       num_filters=50,
                       kernel_size=(2, 2, 2),
                       regularization_constant=.1)
    SOLVER.load_model_weights('data/models/3_shuffle_large_run/weights')


@app.route("/", methods=['GET'])
def index():
    CUBE = Cube()
    return render_template('index.html')


@app.route('/background_process')
def background_process():

    move = request.args.get('move', None, type=str)
    process_move(move, True)
    return "Nothing"



@app.route('/cube_solving', methods=['GET'])
def cube_solving():
    global CUBE
    max_time_steps = 5
    _, CUBE, solver_steps = SOLVER.solve(CUBE, max_time_steps, verbose=False)
    print(CUBE.state)
    print(solver_steps)
    return json.dumps({'moves':solver_steps});

def process_move(move, verbose=False):
    '''
    Alters state of back-end cube based on front-end action

    Parameters:
    -----------
    move : str
        value from front end
    verbose : boolean
    '''
    function_mappings = {'Up': CUBE.up,
                         'Up_p': CUBE.up_p,
                         'Down': CUBE.down,
                         'Down_p': CUBE.down_p,
                         'Left': CUBE.left,
                         'Left_p': CUBE.left_p,
                         'Right': CUBE.right,
                         'Right_p': CUBE.right_p,
                         'Front': CUBE.front,
                         'Front_p': CUBE.front_p,
                         'Back': CUBE.back,
                         'Back_p': CUBE.back_p,
                        }
    inner_move = function_mappings[move]
    inner_move()
    if verbose:
        print(CUBE.state)
    pass


if __name__ == '__main__':
    app.run(debug=True)
