# Import required modules
from Simulate import simulate_one
import os
if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input_solo.json")
    simulate_one(path, disp=True)