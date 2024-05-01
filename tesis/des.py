import numpy as np
from tabulate import tabulate
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import secrets
import matplotlib.pyplot as plt
import concurrent.futures


class DES:
    def __init__(self, key):
        self.key = DES.hex_to_bin(key)
    sboxes = None

    @staticmethod
    def generate_key(key_length):
        if key_length % 8 != 0:
            raise ValueError(
                "La longitud de la clave debe ser un múltiplo de 8")

        num_bytes = key_length // 8
        key = secrets.token_bytes(num_bytes)
        return key.hex()

    @staticmethod
    def hex_to_bin(s):
        mp = {'0': "0000", '1': "0001", '2': "0010", '3': "0011",
              '4': "0100", '5': "0101", '6': "0110", '7': "0111",
              '8': "1000", '9': "1001", 'A': "1010", 'B': "1011",
              'C': "1100", 'D': "1101", 'E': "1110", 'F': "1111"}

        binary = ""
        for char in s:
            # Convertir a mayúsculas antes de buscar en el diccionario
            binary += mp[char.upper()]
        return binary

    @staticmethod
    def get_expansion_d_box():
        """
        Retorna la tabla de expansión D utilizada en DES.

        Returns:
        list: La tabla de expansión D.
        """
        expansion_d_box = [
            32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32, 1
        ]
        return expansion_d_box

    @staticmethod
    def get_initial_permutation_table():
        """
        Retorna la tabla de permutación inicial utilizada en DES.

        Returns:
        list: La tabla de permutación inicial.
        """
        initial_permutation_table = [
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7
        ]
        return initial_permutation_table

    @staticmethod
    def get_final_permutation_table():
        """
        Retorna la tabla de permutación final utilizada en DES.

        Returns:
        list: La tabla de permutación final.
        """
        final_permutation_table = [
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25
        ]
        return final_permutation_table

    @staticmethod
    def get_permutation_box():
        """
        Retorna la tabla de permutación utilizada en la etapa de sustitución (S-box) en DES.

        Returns:
        list: La tabla de permutación para la etapa de sustitución.
        """
        permutation_box = [
            16, 7, 20, 21, 29, 12, 28, 17,
            1, 15, 23, 26, 5, 18, 31, 10,
            2, 8, 24, 14, 32, 27, 3, 9,
            19, 13, 30, 6, 22, 11, 4, 25
        ]
        return permutation_box

    def get_tables():
        initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
                        60, 52, 44, 36, 28, 20, 12, 4,
                        62, 54, 46, 38, 30, 22, 14, 6,
                        64, 56, 48, 40, 32, 24, 16, 8,
                        57, 49, 41, 33, 25, 17, 9, 1,
                        59, 51, 43, 35, 27, 19, 11, 3,
                        61, 53, 45, 37, 29, 21, 13, 5,
                        63, 55, 47, 39, 31, 23, 15, 7]

        exp_d = [
            32, 1, 2, 3, 4, 5, 4, 5,
            6, 7, 8, 9, 8, 9, 10, 11,
            12, 13, 12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21, 20, 21,
            22, 23, 24, 25, 24, 25, 26, 27,
            28, 29, 28, 29, 30, 31, 32, 1
        ]

        per = [
            16, 7, 20, 21, 29, 12, 28, 17,
            1, 15, 23, 26, 5, 18, 31, 10,
            2, 8, 24, 14, 32, 27, 3, 9,
            19, 13, 30, 6, 22, 11, 4, 25
        ]

        final_perm = [
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25
        ]
        return initial_perm, exp_d, per, final_perm

    def generate_round_keys(self):
        # Permuted Choice 1 (PC1) table

        pc1_table = [
            57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4
        ]
        # Permuted Choice 2 (PC2) table
        pc2_table = [
            14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32
        ]

        # Left shift schedule for each round
        left_shift_schedule = [
            1, 1, 2, 2,
            2, 2, 2, 2,
            1, 2, 2, 2,
            2, 2, 2, 1
        ]

        key_permuted = [self.key[index - 1] for index in pc1_table]

        # Split the key into left and right halves
        left_half = key_permuted[:28]
        right_half = key_permuted[28:]

        round_keys = []

        # Generate round keys
        for i in range(16):
            # Perform left circular shift on left and right halves
            left_half = left_half[left_shift_schedule[i]                                  :] + left_half[:left_shift_schedule[i]]
            right_half = right_half[left_shift_schedule[i]                                    :] + right_half[:left_shift_schedule[i]]

            # Combine left and right halves
            combined_halves = left_half + right_half

            # Apply PC2 permutation to obtain round key
            round_key = [combined_halves[index - 1] for index in pc2_table]

            # Add round key to list of round keys
            round_keys.append(round_key)

        return round_keys

    @staticmethod
    def bin_to_hex(s):
        mp = {
            "0000": '0', "0001": '1', "0010": '2', "0011": '3',
            "0100": '4', "0101": '5', "0110": '6', "0111": '7',
            "1000": '8', "1001": '9', "1010": 'A', "1011": 'B',
            "1100": 'C', "1101": 'D', "1110": 'E', "1111": 'F'
        }

        hex_str = ""
        # Divide la cadena binaria en segmentos de 4 bits y los convierte a hexadecimal
        for i in range(0, len(s), 4):
            bin_chunk = s[i:i+4]
            hex_str += mp[bin_chunk]
        return hex_str

    @staticmethod
    def bin_to_dec(binary):
        decimal, i = 0, 0
        while binary != 0:
            dec = binary % 10
            decimal += dec * pow(2, i)
            binary //= 10
            i += 1
        return decimal

    @staticmethod
    def dec_to_bin(num):
        res = bin(num).replace("0b", "")
        res = res.zfill(4 * ((len(res) + 3) // 4))
        return res

    @staticmethod
    def permute(k, arr, n):
        permutation = ""
        for i in range(0, n):
            permutation = permutation + k[arr[i] - 1]
        return permutation

    @staticmethod
    def shift_left(k, nth_shifts):
        return k[nth_shifts:] + k[:nth_shifts]

    @staticmethod
    def xor(a, b):
        return ''.join('0' if x == y else '1' for x, y in zip(a, b))

    @staticmethod
    def generate_round_keys(shift_table, key_comp, left, right, left_with_AI, right_with_AI):
        rkb_without_AI = []
        rk_without_AI = []
        rkb_with_AI = []
        rk_with_AI = []

        # Generate round keys for DES without AI
        for i in range(0, 16):
            left = DES.shift_left(left, shift_table[i])
            right = DES.shift_left(right, shift_table[i])

            combine_str = left + right
            round_key = DES.permute(combine_str, key_comp, 48)

            rkb_without_AI.append(round_key)
            rk_without_AI.append(DES.bin_to_hex(round_key))

        # Generate round keys for DES with AI
        for i in range(0, 16):
            left_with_AI = DES.shift_left(left_with_AI, shift_table[i])
            right_with_AI = DES.shift_left(right_with_AI, shift_table[i])

            combine_str_with_AI = left_with_AI + right_with_AI
            round_key_with_AI = DES.permute(combine_str_with_AI, key_comp, 48)

            rkb_with_AI.append(round_key_with_AI)
            rk_with_AI.append(DES.bin_to_hex(round_key_with_AI))

        return rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI


class Encryption(DES):
    def __init__(self, key):
        super().__init__(key)

    def encrypt(pt, rkb, rk, sbox):
        # Convert plaintext to binary
        pt_bin = DES.hex_to_bin(pt)

        # Initial Permutation
        pt_permuted = DES.permute(
            pt_bin, DES.get_initial_permutation_table(), 64)

        # Split the permuted plaintext into left and right halves
        left, right = pt_permuted[:32], pt_permuted[32:]

        # Encryption rounds
        for i in range(16):
            # Expansion D-box
            right_expanded = DES.permute(right, DES.get_expansion_d_box(), 48)

            # XOR with round key
            xor_result = DES.xor(right_expanded, rkb[i])

            # S-box substitution
            sbox_str = ""
            for j in range(8):
                row = DES.bin_to_dec(
                    int(xor_result[j * 6] + xor_result[j * 6 + 5]))
                col = DES.bin_to_dec(int(xor_result[j * 6 + 1:j * 6 + 5]))
                val = sbox[j][row][col]
                sbox_str += DES.dec_to_bin(val)

            # Straight D-box
            straight_permuted = DES.permute(
                sbox_str, DES.get_permutation_box(), 32)

            # XOR with left
            result = DES.xor(left, straight_permuted)
            left = right
            right = result

        # Combination
        combined = right + left

        # Final permutation
        cipher_text = DES.permute(
            combined, DES.get_final_permutation_table(), 64)
        return cipher_text


class Decryption(DES):
    def __init__(self, key):
        super().__init__(key)

    @staticmethod
    def decrypt(cipher_text, rkb, rk, sbox):
        rkb_rev = rkb[::-1]
        rk_rev = rk[::-1]
        text = DES.bin_to_hex(Encryption.encrypt(
            cipher_text, rkb_rev, rk_rev, sbox))
        return text


keyp = [
    57, 49, 41, 33, 25, 17, 9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 63, 55, 47, 39,
    31, 23, 15, 7, 62, 54, 46, 38,
    30, 22, 14, 6, 61, 53, 45, 37,
    29, 21, 13, 5, 28, 20, 12, 4
]


class AIMODEL:
    def __init__(self):
        pass

    @staticmethod
    def generate_sboxes(num_sboxes, rows=4, cols=16):
        if DES.sboxes is None:
            DES.sboxes = []
        for _ in range(num_sboxes):
            sbox = np.arange(0, rows * cols)
            np.random.shuffle(sbox)
            if len(sbox) < rows * cols:
                repetitions = (rows * cols - len(sbox)) // len(sbox) + 1
                sbox = np.tile(sbox, repetitions)[:rows * cols]
            sbox = np.reshape(sbox, (rows, cols))
            # Round the values to the nearest integer
            sbox = np.round(sbox).astype(int)
            DES.sboxes.append(sbox)
        return DES.sboxes

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
    def linearity(sbox):
        n = int(np.log2(len(sbox)))
        max_abs_correlation = 0

        total_iterations = (2 ** n - 1) ** 2  # Total de iteraciones
        completed_iterations = 0

        for input_mask in range(1, 2 ** n):
            for output_mask in range(1, 2 ** n):
                correlation = 0
                sum_ = 0
                for input_val in range(0, 2 ** n):
                    input_val_xor_input_mask = input_val ^ (
                        input_val & input_mask)
                    output_val = sbox[input_val]
                    if isinstance(output_val, np.ndarray):
                        # Sum all elements of the array
                        output_val_scalar = np.sum(output_val)
                    else:
                        output_val_scalar = output_val
                    output_val_xor_output_mask = output_val_scalar ^ (
                        output_val_scalar & output_mask)
                    sum_ += bin(input_val_xor_input_mask).count(
                        '1') ^ bin(output_val_xor_output_mask).count('1')
                correlation += (-1) ** bin(sum_).count('1')
                correlation = abs(correlation / (2 ** n))
                max_abs_correlation = max(max_abs_correlation, correlation)

                completed_iterations += 1
                progress_percentage = (
                    completed_iterations / total_iterations) * 100
                print(
                    f"\rProgreso: {progress_percentage:.2f}%", end='', flush=True)

        print()  # Imprimir una nueva línea al final para que la salida sea limpia

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

        # Ensure dlct_score has shape (100, 1)
        dlct_score = np.array([dlct_score] * 100).reshape(-1, 1)

        out["F"] = dlct_score

    def _calc_pbounds(self, **kwargs):
        if self.xl is not None and self.xu is not None:
            xl = self.xl
            xu = self.xu
        else:
            xl = np.zeros(self.n_var)
            xu = np.ones(self.n_var)
        return xl, xu


def main():
    pt = "ADC0326456789BAEF"
    print(" Before the encription Plain Text : ", pt)
    num_sboxes = 64
    # Key generation for DES algorithm without AI
    key_without_AI = DES.generate_key(key_length=128)
    key_without_AI = DES.hex_to_bin(key_without_AI)

    key_without_AI = DES.permute(key_without_AI, keyp, 56)

    # Key generation for DES algorithm with AI
    key_with_AI = DES.generate_key(key_length=128)
    key_with_AI = DES.hex_to_bin(key_with_AI)
    key_with_AI = DES.permute(key_with_AI, keyp, 56)

    shift_table = [1, 1, 2, 2,
                   2, 2, 2, 2,
                   1, 2, 2, 2,
                   2, 2, 2, 1]

    key_comp = [14, 17, 11, 24, 1, 5,
                3, 28, 15, 6, 21, 10,
                23, 19, 12, 4, 26, 8,
                16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55,
                30, 40, 51, 45, 33, 48,
                44, 49, 39, 56, 34, 53,
                46, 42, 50, 36, 29, 32]

    left = key_without_AI[0:28]
    right = key_without_AI[28:56]

    rkb_without_AI = []
    rk_without_AI = []

    left_with_AI = key_with_AI[0:28]
    right_with_AI = key_with_AI[28:56]

    rkb_with_AI = []
    rk_with_AI = []

    rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI = DES.generate_round_keys(
        shift_table, key_comp, left, right, left_with_AI, right_with_AI)

    # DES without AI
    
    rows = 8  # Número de filas de las S-boxes
    cols = 16  # Número de columnas de las S-boxes
    sboxes = AIMODEL.generate_sboxes(num_sboxes, rows, cols)

    print("Encryption without AI")
    cipher_text_without_AI = DES.bin_to_hex(
        Encryption.encrypt(pt, rkb_without_AI, rk_without_AI, sboxes))
    print("Cipher Text without AI: ", cipher_text_without_AI)

    print("Decryption without AI")
    plain_text_without_AI = Decryption.decrypt(
        cipher_text_without_AI, rkb_without_AI, rk_without_AI, sboxes)
    print("Plain Text without AI: ", plain_text_without_AI)

    linearity_values_without_AI = AIMODEL.linearity(sboxes)

    print("Correlación de linealidad de las S-boxes:",
          linearity_values_without_AI)

    
    # DES with AI
    
    population_size = 100
    best_sboxes, best_score = AIMODEL.geneticModel(num_sboxes, population_size)

    print("S-box found:" + str(best_sboxes))



    print("Best S-box found:")
    # print(best_sboxes.reshape(num_sboxes, 4, 16))
    print("DLCT score of the best S-box:", best_score)

    print("Encryption with AI")
    cipher_text_with_AI = DES.bin_to_hex(Encryption.encrypt(
        pt, rkb_with_AI, rk_with_AI, best_sboxes.reshape(num_sboxes, 4, 16)))
    print("Cipher Text with AI: ", cipher_text_with_AI)

    print("Decryption with AI")
    plain_text_with_AI = Decryption.decrypt(
        cipher_text_with_AI, rkb_with_AI, rk_with_AI, best_sboxes.reshape(num_sboxes, 4, 16))
    print("Decrypt result text with AI: ", plain_text_with_AI)

    diff_table = AIMODEL.differential_table(best_sboxes)
    lin_table = AIMODEL.linear_table(best_sboxes)
    linearity_values = AIMODEL.linearity(best_sboxes)


    print("Correlación de linealidad de las S-boxes:", linearity_values)

    # Graficar la tabla de diferencias
    AIMODEL.plot_table(diff_table, 'Tabla de Diferencias con IA')
    
    # Graficar la tabla de linealidad
    AIMODEL.plot_table(lin_table, 'Tabla de Linealidad con IA')


if __name__ == "__main__":
    main()
