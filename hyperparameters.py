from integration_task import Integration_Task
from flipflop_task import Flipflop_task
# RNN
LEARNING_RATE = 0.001


# Fix Point Finder
CANDIDATES_BATCHSIZE = 100
CANDIDATES_ITERS = 50
#amount of of sampled possible pre-candidates, num_points must be smaller than this.
CANDIDATES_SAMPLED=CANDIDATES_BATCHSIZE*CANDIDATES_ITERS
P_NORM=2
LEARNING_RATE = 0.001
OPTIMIZATION_ITERS = 1000
PRINT_EVERY = 100
#TASK_CLASS = Integration_Task
TASK_CLASS = Flipflop_task
# Fix Points postprocessing
FP_TOL = 1
FP_OPT_STOP_TOL = 0.001
OUTLIER_TOL = 1.0
UNIQUE_TOL = 0.0025


#MODEL_PATH = 'models/GRU_integration_09-08-2020_03-13-31_PM/'
MODEL_PATH = 'models/GRU_flipflop_19-08-2020_06-56-11_PM/'
#MODEL_NAME = 'trained_weights_GRU_integration_epochs_5'
MODEL_NAME = 'trained_weights_GRU_flipflop_epochs_5'
#FIX_POINT_FILE = ['fixpoints_and_losses_input_negative19-08-2020_06-03-41_PM', 'fixpoints_and_losses_input_positive19-08-2020_06-03-11_PM', 'fixpoints_and_losses_input_zeros19-08-2020_06-02-55_PM']
FIX_POINT_FILE = ['fixpoints_and_losses_input_zeros19-08-2020_07-15-39_PM']
#RUN_FILE = ['increasing_runs_19-08-2020_06-03-41_PM']# ['exemplary_runs_19-08-2020_06-02-16_PM' ]##['decreasing_runs_19-08-2020_06-03-41_PM']#, #,
RUN_FILE = []
