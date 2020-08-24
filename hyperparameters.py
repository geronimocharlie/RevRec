from integration_task import Integration_Task
from flipflop_task import Flipflop_task
import os

current_dir = os.getcwd()

mode = 'charlie'

if mode == 'charlie':

    # Task
    TASK_CLASS = Integration_Task
    INPUT_SIZE = 1
    OUTPUT_SIZE = 1
    LENGHT = 100
    # model parameters
    LEARNING_RATE = 0.001
    HIDDEN_SIZE = 100
    NUM_LAYERS = 1
    BATCH_SIZE = 10
    NUM_EPOCHS = 5

    # Fix Point Finder
    NUM_POINTS = 10000
    CANDIDATES_BATCHSIZE = 100
    CANDIDATES_ITERS = 50
    # amount of of sampled possible pre-candidates, num_points must be smaller than this.
    CANDIDATES_SAMPLED = CANDIDATES_BATCHSIZE * CANDIDATES_ITERS
    P_NORM = 2
    LEARNING_RATE = 0.001
    OPTIMIZATION_ITERS = 1000
    PRINT_EVERY = 100
    TASK_CLASS = Integration_Task

    # Fix Points postprocessing
    FP_TOL = 1
    FP_OPT_STOP_TOL = 0.001
    OUTLIER_TOL = 1.0
    UNIQUE_TOL = 0.0025

    MODEL_PATH = os.path.join(
        current_dir, 'models/GRU_integration_23-08-2020_11-44-51_AM/')
    MODEL_NAME = 'trained_weights_GRU_integration_epochs_5'
    FIX_POINT_FILE = ['fixpoints_and_losses_input_negative23-08-2020_11-58-16_AM',
                      'fixpoints_and_losses_input_positive23-08-2020_11-57-34_AM', 'fixpoints_and_losses_input_zeros23-08-2020_11-57-11_AM']
    RUN_FILE = []  # 'increasing_runs_23-08-2020_11-58-16_AM', 'decreasing_runs_23-08-2020_11-58-16_AM', 'exemplary_runs_23-08-2020_11-56-10_AM' # ['increasing_runs_19-08-2020_06-03-41_PM']# ['exemplary_runs_19-08-2020_06-02-16_PM' ]##['decreasing_runs_19-08-2020_06-03-41_PM']#, #,

elif mode == 'leon':
    TASK_CLASS = Flipflop_task
    INPUT_SIZE = 3
    OUTPUT_SIZE = 3
    LENGHT = 100
    # RNN
    LEARNING_RATE = 0.001
    # define task and model parameters
    HIDDEN_SIZE = 100
    NUM_LAYERS = 1
    BATCH_SIZE = 10
    NUM_EPOCHS = 5

    # Fix Point Finder
    NUM_POINTS = 10000
    CANDIDATES_BATCHSIZE = 100
    CANDIDATES_ITERS = 50
    # amount of of sampled possible pre-candidates, num_points must be smaller than this.
    CANDIDATES_SAMPLED = CANDIDATES_BATCHSIZE * CANDIDATES_ITERS
    P_NORM = 2
    LEARNING_RATE = 0.001
    OPTIMIZATION_ITERS = 1000
    PRINT_EVERY = 100
    TASK_CLASS = Integration_Task

    # Fix Points postprocessing
    FP_TOL = 1
    FP_OPT_STOP_TOL = 0.001
    OUTLIER_TOL = 1.0
    UNIQUE_TOL = 0.0025

    MODEL_PATH = ''
    MODEL_NAME = ''
    FIX_POINT_FILE = []
    RUN_FILE = []
