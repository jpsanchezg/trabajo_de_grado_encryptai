from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import numpy as np
import random

import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import differential_evolution
class AI_MODEL:
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
            initial_sbox = list(range(rows * cols))
            random.shuffle(initial_sbox)
            result = differential_evolution(AI_MODEL.sbox_cost, [(0, rows*cols-1)]*(rows*cols),
                                            args=(), tol=1e-5, maxiter=100, popsize=15, mutation=(0.5, 1),
                                            recombination=0.7, seed=None, callback=None, disp=False,
                                            polish=True, init='latinhypercube', atol=0)
            optimized_sbox = np.reshape(result.x, (rows, cols)).tolist()
            sboxes.append(optimized_sbox)
        return sboxes

    
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
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.num_variables, activation='relu'))
        model.add(Dense(32, activation='relu'))
        # Output layer, linear activation
        model.add(Dense(32, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def state_to_input(self, state):
        return np.eye(32)[state].flatten()  # One-hot encode the state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 31)  # Explore: select a random action
        else:
            state_input = self.state_to_input(state)
            q_values = self.model.predict(state_input.reshape(1, -1))[0]
            # Exploit: select action with max Q-value
            return np.argmax(q_values)

    def learn(self, current_state, action, reward, next_state):
        current_state_input = self.state_to_input(current_state)
        next_state_input = self.state_to_input(next_state)

        target = reward + self.discount_factor * \
            np.amax(self.model.predict(next_state_input.reshape(1, -1))[0])
        target_full = self.model.predict(current_state_input.reshape(1, -1))[0]
        target_full[action] = target
        self.model.fit(current_state_input.reshape(1, -1),
                       target_full.reshape(-1, 32), epochs=1, verbose=0)

    def objective_function(self, sbox):
        # Placeholder for the actual objective function
        # Calculate properties such as nonlinearity, differential uniformity, etc.
        return np.random.random()  # Example: random reward

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

                # Placeholder condition to end an episode
                if np.random.random() < 0.1:
                    done = True

        return best_solution, best_score
