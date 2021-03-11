import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, help='The number of tasks')
args = parser.parse_args()
task_number: int = args.task

np.savetxt("./project/results/test"+str(task_number)+".txt",np.random.normal(5, 1, 5))