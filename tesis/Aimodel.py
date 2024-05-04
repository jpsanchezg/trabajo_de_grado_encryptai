from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import random

# esto es para el uso de todo el procesador para el cálculo de la correlación
import concurrent.futures
from tqdm import tqdm
from functools import partial

from scipy.optimize import differential_evolution
class AIMODEL:
    def __init__(self):
        pass
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
            result = differential_evolution(AIMODEL.sbox_cost, [(0, rows*cols-1)]*(rows*cols),
                                            args=(), tol=1e-5, maxiter=100, popsize=15, mutation=(0.5, 1),
                                            recombination=0.7, seed=None, callback=None, disp=False,
                                            polish=True, init='latinhypercube', atol=0)
            # Reshape the resulting permutation into rows and columns
            optimized_sbox = np.reshape(result.x, (rows, cols)).tolist()
            sboxes.append(optimized_sbox)
        return sboxes

    @staticmethod
    def differential_table(sbox):
        table = {}
        for input_diff in range(16):
            for output_diff in range(16):
                xor_func = np.vectorize(lambda x: x ^ input_diff)
                count = np.sum(sbox ^ xor_func(sbox) == output_diff)
                table[(input_diff, output_diff)] = count
        return table

    @staticmethod
    def linear_table(sboxes):
        table = {}
        for input_mask in range(16):
            for output_mask in range(16):
                count = 0
                for input_val in range(16):
                    sbox_row = sboxes[input_val]
                    # Use np.nditer to iterate over elements of the numpy array
                    for sbox_val in np.nditer(sbox_row):
                        if bin(input_val & input_mask).count('1') ^ bin(sbox_val & output_mask).count('1') == 0:
                            count += 1
                table[(input_mask, output_mask)] = count
        return table

    @staticmethod
    def compute_correlation(input_mask, output_mask, sbox):
        correlation = 0
        sum_ = 0
        n = int(np.log2(len(sbox)))
        input_mask = int(input_mask)  # Convertir a entero si no lo es
        output_mask = int(output_mask)  # Convertir a entero si no lo es
        for input_val in range(0, 2 ** n):
            input_val_xor_input_mask = input_val ^ (input_val & input_mask)
            output_val = sbox[input_val]
            if isinstance(output_val, np.ndarray):
                output_val_scalar = np.sum(output_val)
            else:
                output_val_scalar = output_val
            if isinstance(output_val_scalar, np.ndarray):
                # Convertir a escalar si es un array de numpy
                output_val_scalar = output_val_scalar.item()
            # Convertir a entero después de comprobaciones
            output_val_scalar = int(output_val_scalar)
            output_val_xor_output_mask = output_val_scalar ^ (
                output_val_scalar & output_mask)
            sum_ += bin(input_val_xor_input_mask).count(
                '1') ^ bin(output_val_xor_output_mask).count('1')
        correlation += (-1) ** bin(sum_).count('1')
        correlation = abs(correlation / (2 ** n))
        return correlation

    @staticmethod
    def linearity(sbox):
        n = int(np.log2(len(sbox)))
        max_abs_correlation = 0

        total_iterations = (2 ** n - 1) ** 2  # Total de iteraciones
        chunk_size = 4000  # Ajusta este valor según tus restricciones de memoria

        chunks = []
        for input_mask in range(1, 2 ** n):
            for output_mask in range(1, 2 ** n):
                chunks.append((input_mask, output_mask))

        with concurrent.futures.ProcessPoolExecutor() as executor, tqdm(total=total_iterations) as pbar:
            for i in range(0, len(chunks), chunk_size):
                chunk = chunks[i:i+chunk_size]
                compute_partial = partial(AIMODEL.compute_correlation, sbox=sbox)
                futures = [executor.submit(compute_partial, *params)
                           for params in chunk]
                for future in concurrent.futures.as_completed(futures):
                    correlation = future.result()
                    max_abs_correlation = max(max_abs_correlation, correlation)
                    pbar.update(len(chunk))

        return 1 - 2 * max_abs_correlation

    @staticmethod
    def geneticModel(num_sboxes, population_size):
        num_variables = num_sboxes * 64
        lower_bound = 0
        upper_bound = 31  # Upper bound set to the maximum value in the sbox

        # Array filled with lower_bound
        xl = np.full(num_variables, lower_bound)
        # Array filled with upper_bound
        xu = np.full(num_variables, upper_bound)
        problem = SboxProblem(num_sboxes, xl=xl, xu=xu)
        algorithm = GA(pop_size=population_size)

        res = minimize(problem, algorithm, seed=1, verbose=False)

        best_solution = np.round(res.X).astype(
            int)  # Round the solution to integers
        best_score = res.F[0]

        return best_solution, best_score

    def plot_table(table, title):
        fig, ax = plt.subplots()
        ax.bar(range(len(table)), list(table.values()), align='center')
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table.keys(), rotation=90)
        ax.set_xlabel('Diferencia de entrada/salida')
        ax.set_ylabel('Recuento')
        ax.set_title(title)
        plt.show()

    @staticmethod
    def calculate_dlct_score(sboxes):
        dlct_score_total = 0
        for sbox in sboxes:
            dlct_score_sbox = 0
            n = len(sbox)
            sbox = sbox.astype(int)
            if sbox.ndim != 2:
                raise ValueError("S-boxes must be 2-dimensional arrays.")
            for alpha in range(n):
                count_diff = np.sum(np.bitwise_xor(
                    sbox, np.roll(sbox, alpha, axis=0))[:, :, np.newaxis] == np.arange(n))
                dlct_score_sbox += np.abs(count_diff - (n / 2)).sum()
            dlct_score_total += dlct_score_sbox
        return dlct_score_total

    def calculate_differential_uniformity(sbox):
        min_differential_probability = float('inf')
        for input_diff in range(0, 16):
            for output_diff in range(0, 16):
                if input_diff == 0 and output_diff == 0:
                    continue
                count = 0
                for sbox_row in sbox:
                    for x in range(16):
                        # Convert each element of the sbox row to integer before XOR operation
                        sbox_x = int(sbox_row[x])
                        sbox_x_diff = int(sbox_row[(x ^ input_diff) % 16])
                        if np.bitwise_xor(sbox_x, sbox_x_diff) == output_diff:
                            count += 1
                probability = count / (16 * len(sbox))
                if probability < min_differential_probability:
                    min_differential_probability = probability
        return min_differential_probability


class SboxProblem(Problem):
    def __init__(self, num_sboxes, xl=None, xu=None):
        super().__init__(n_var=num_sboxes * 64, n_obj=1,
                         n_constr=0, elementwise_evaluation=True)
        self.num_sboxes = num_sboxes
        self.xl = xl
        self.xu = xu

    def _evaluate(self, X, out, *args, **kwargs):
        sboxes = np.array_split(X, self.num_sboxes)
        dlct_score = AIMODEL.calculate_dlct_score(sboxes)
        differential_uniformity_score = np.mean(
            [AIMODEL.calculate_differential_uniformity(sbox) for sbox in sboxes])

        # Adjust weights according to importance
        total_score = dlct_score * 0.6 + differential_uniformity_score * 0.4

        # Ensure total_score has shape (100, 1)
        total_score = np.array([total_score] * 100).reshape(-1, 1)

        out["F"] = total_score

    def _calc_pbounds(self, **kwargs):
        if self.xl is not None and self.xu is not None:
            xl = self.xl
            xu = self.xu
        else:
            xl = np.zeros(self.n_var)
            xu = np.ones(self.n_var)
        return xl, xu
