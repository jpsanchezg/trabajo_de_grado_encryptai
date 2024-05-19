from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
import random
from scipy.optimize import differential_evolution, dual_annealing

from analisis import Test
import torch
import torch.nn as nn
import torch.optim as optim

class AI_MODEL:
    def __init__(self):
        pass

    @staticmethod
    def sbox_cost(sbox, rows=8, cols=64):  # Corrected signature
        correlation = np.corrcoef(sbox.flatten(), sbox.flatten())[0, 1]
        cost_correlation = -abs(correlation)

        average_sac = Test.calculate_average_sac(sbox)
        cost_sac = -(average_sac - 0.4) ** 2

        total_cost = 0.3 * cost_correlation + 0.7 * cost_sac
        return total_cost

    @staticmethod
    def generate_sboxes_differential_evolution(num_sboxes, rows=4, cols=16):
        sboxes = []
        bounds = [(0, 31) for _ in range(rows * cols)]
        for _ in range(num_sboxes):
            initial_sbox = list(range(rows * cols))
            random.shuffle(initial_sbox)
            result = differential_evolution(
                AI_MODEL.sbox_cost,
                bounds,
                args=(rows, cols),  # Pass arguments as a tuple
                maxiter=100,
                popsize=15,
                disp=False,
                polish=True,
            )
            optimized_sbox = np.reshape(result.x, (rows, cols)).tolist()
            sboxes.append(optimized_sbox)
        return sboxes

    @staticmethod
    def generate_sboxes_SA(num_sboxes, rows=4, cols=16):
        sboxes = []
        bounds = [(0, 31) for _ in range(rows * cols)]
        for _ in range(num_sboxes):
            initial_sbox = list(range(rows * cols))
            random.shuffle(initial_sbox)

            result = dual_annealing(
                AI_MODEL.sbox_cost, bounds, x0=initial_sbox, maxiter=1000
            )
            optimized_sbox = np.reshape(result.x, (rows, cols)).tolist()
            sboxes.append(optimized_sbox)
        return sboxes
    @staticmethod
    def geneticModel(num_sboxes, rows=4, cols=16, population_size=100, initial_sboxes=None):
        num_variables = num_sboxes * 64
        lower_bound = 0
        upper_bound = 31

        xl = np.full(num_variables, lower_bound)
        xu = np.full(num_variables, upper_bound)
        problem = SboxProblem(num_sboxes, xl=xl, xu=xu)
        algorithm = GA(pop_size=population_size)

        res = minimize(problem, algorithm, seed=1, verbose=False)

        best_solution = np.round(res.X).astype(int)
        best_score = res.F[0]

        return best_solution, best_score

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

    @staticmethod
    def calculate_differential_uniformity(sbox):
        min_differential_probability = float('inf')
        for input_diff in range(0, 16):
            for output_diff in range(0, 16):
                if input_diff == 0 and output_diff == 0:
                    continue
                count = 0
                for sbox_row in sbox:
                    for x in range(16):
                        sbox_x = int(sbox_row[x])
                        sbox_x_diff = int(sbox_row[(x ^ input_diff) % 16])
                        if np.bitwise_xor(sbox_x, sbox_x_diff) == output_diff:
                            count += 1
                probability = count / (16 * len(sbox))
                if probability < min_differential_probability:
                    min_differential_probability = probability
        return min_differential_probability

    @staticmethod
    def generate_sboxes_combined(num_sboxes, rows=4, cols=16):
        de_sboxes = AI_MODEL.generate_sboxes_differential_evolution(
            num_sboxes, rows, cols)
        sa_sboxes = AI_MODEL.generate_sboxes_SA(num_sboxes, rows, cols)

        return de_sboxes, sa_sboxes
    @staticmethod
    def geneticModel_combined(num_sboxes, rows=4, cols=16, population_size=100, initial_sboxes=None):
        de_sboxes, sa_sboxes = AI_MODEL.generate_sboxes_combined(
            num_sboxes, rows, cols)

        combined_sboxes = de_sboxes + sa_sboxes  # Combine both sets

        # Pass combined_sboxes as initial_sboxes (if applicable)
        return AI_MODEL.geneticModel(num_sboxes, rows, cols, population_size, initial_sboxes=combined_sboxes)



class SboxProblem(Problem):
    def __init__(self, num_sboxes, xl=None, xu=None):
        super().__init__(n_var=num_sboxes * 64, n_obj=1,
                         n_constr=0, elementwise_evaluation=True)
        self.num_sboxes = num_sboxes
        self.xl = xl
        self.xu = xu

    def _evaluate(self, X, out, *args, **kwargs):
        sboxes = np.array_split(X, self.num_sboxes)
        dlct_score = AI_MODEL.calculate_dlct_score(sboxes)
        differential_uniformity_score = np.mean(
            [AI_MODEL.calculate_differential_uniformity(sbox) for sbox in sboxes])

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


class QLearningAgent:
    def __init__(self, num_sboxes, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, episodes=10000):

        self.num_sboxes = num_sboxes
        self.num_variables = num_sboxes * 64
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def build_model(self):
        
        return nn.Sequential(
            nn.Linear(self.num_variables * 32, 64),  # Capa de entrada
            nn.ReLU(),
            nn.Linear(64, 32),  # Capa oculta
            nn.ReLU(),
            nn.Linear(32, 32)  # Capa de salida
        )

    def state_to_input(self, state):
        
        return torch.tensor(np.eye(32)[state].flatten(), dtype=torch.float32, device=self.device)

    def choose_action(self, state):
        
        if random.uniform(0, 1) < self.epsilon:
            # Explorar: seleccionar una acción aleatoria
            return random.randint(0, 31)
        else:
            state_input = self.state_to_input(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_input)
            # Explotar: seleccionar la acción con el valor Q máximo
            return q_values.argmax().item()

    def learn(self, current_state, action, reward, next_state):
        
        current_state_input = self.state_to_input(current_state).unsqueeze(0)
        next_state_input = self.state_to_input(next_state).unsqueeze(0)

        self.optimizer.zero_grad()

        current_q_values = self.model(current_state_input)
        next_q_values = self.model(next_state_input)

        target = reward + self.discount_factor * \
            torch.max(next_q_values).item()
        target_f = current_q_values.clone()
        target_f[0, action] = target

        loss = self.loss_fn(current_q_values, target_f)
        loss.backward()
        self.optimizer.step()

    def objective_function(self, sbox):
        
        return np.random.random()  # Ejemplo: recompensa aleatoria

    def train(self):
        best_solution = None
        best_score = float('-inf')

        for episode in range(self.episodes):
            current_state = np.random.randint(0, 32, self.num_variables)
            done = False
            while not done:
                action = self.choose_action(current_state)
                next_state = current_state.copy()
                next_state[random.randint(0, self.num_variables - 1)] = action

                reward = self.objective_function(next_state)
                self.learn(current_state, action, reward, next_state)

                if reward > best_score:
                    best_score = reward
                    best_solution = next_state.copy()

                current_state = next_state

                # Condición de finalización de un episodio
                if np.random.random() < 0.1:
                    done = True

        return best_solution, best_score
