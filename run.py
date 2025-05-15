# Import required modules
from Simulate import simulate_one, simulate_all
import os
if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_mult.json")
    simulate_all(path, plot=False, disp=True)