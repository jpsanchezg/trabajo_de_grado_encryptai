import random
import numpy as np
from scipy.optimize import differential_evolution


class SBoxGenerator:
    @staticmethod
    def sbox_cost(sbox):
        # Calculate the linear correlation coefficient
        correlation = np.corrcoef(
            np.array(sbox).flatten(), np.array(sbox).flatten())[0, 1]
        # Cost is the negative of correlation to minimize it
        return -abs(correlation)

    @staticmethod
    def generate_sboxes(num_sboxes, rows=4, cols=16):
        sboxes = []
        for _ in range(num_sboxes):
            # Initialize S-box with a random permutation
            initial_sbox = list(range(rows * cols))
            random.shuffle(initial_sbox)
            # Minimize correlation using Differential Evolutionary Algorithm
            result = differential_evolution(SBoxGenerator.sbox_cost, [(0, rows*cols-1)]*(rows*cols),
                                            args=(), tol=1e-5, maxiter=100, popsize=15, mutation=(0.5, 1),
                                            recombination=0.7, seed=None, callback=None, disp=False,
                                            polish=True, init='latinhypercube', atol=0)
            # Reshape the resulting permutation into rows and columns
            optimized_sbox = np.reshape(result.x, (rows, cols)).tolist()
            sboxes.append(optimized_sbox)
        return sboxes


# Example usage:
sboxes = SBoxGenerator.generate_sboxes(4)
for idx, sbox in enumerate(sboxes):
    print(f"S-box {idx + 1}:")
    for row in sbox:
        print(row)
    print()
