from integration_task import Integration_Task
# RNN
LEARNING_RATE = 0.001


# Fix Point Finder
CANDIDATES_BATCHSIZE = 100
CANDIDATES_ITERS = 50
#amount of of sampled possible pre-candidates, num_points must be smaller than this.
CANDIDATES_SAMPLED=CANDIDATES_BATCHSIZE*CANDIDATES_ITERS
P_NORM=2
LEARNING_RATE = 0.001
OPTIMIZATION_ITERS = 5000
PRINT_EVERY = 100
TASK_CLASS = Integration_Task

# Fix Points postprocessing
FP_TOL = 1e-14
FP_OPT_STOP_TOL = 0.00001
OUTLIER_TOL = 1.0
UNIQUE_TOL = 0.0025
