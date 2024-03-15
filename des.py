import numpy as np
from tabulate import tabulate
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.optimize import minimize


class DES:
    def __init__(self, key):
        self.key = DES.hex_to_bin(key)
    sboxes = None

    @staticmethod
    def hex_to_bin(s):
        mp = {'0': "0000", '1': "0001", '2': "0010", '3': "0011",
              '4': "0100", '5': "0101", '6': "0110", '7': "0111",
              '8': "1000", '9': "1001", 'A': "1010", 'B': "1011",
              'C': "1100", 'D': "1101", 'E': "1110", 'F': "1111"}
        binary = ""
        for char in s:
            binary += mp[char]
        return binary

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
        sbox = AIMODEL.generate_sboxes(8)
        return initial_perm, exp_d, per, sbox, final_perm

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
            left_half = left_half[left_shift_schedule[i]:] + left_half[:left_shift_schedule[i]]
            right_half = right_half[left_shift_schedule[i]:] + right_half[:left_shift_schedule[i]]

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
        for i in range(0, len(s), 4):
            ch = s[i:i + 4]
            hex_str += mp[ch]
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


class Encryption(DES):
    def __init__(self, key):
        super().__init__(key)

    @staticmethod
    def encrypt(pt, rkb, rk):
        pt = DES.hex_to_bin(pt)
        initial_perm, exp_d, per, sbox, final_perm = DES.get_tables()

        # Initial Permutation
        pt = DES.permute(pt, initial_perm, 64)
        print("After initial permutation", DES.bin_to_hex(pt))

        # Splitting
        left = pt[0:32]
        right = pt[32:64]
        for i in range(0, 16):
            # Expansion D-box: Expanding the 32 bits data into 48 bits
            right_expanded = DES.permute(right, exp_d, 48)

            # XOR RoundKey[i] and right_expanded
            xor_x = DES.xor(right_expanded, rkb[i])

            # S-boxex: substituting the value from s-box table by calculating row and column
            sbox_str = ""
            for j in range(0, 8):
                row = DES.bin_to_dec(int(xor_x[j * 6] + xor_x[j * 6 + 5]))
                col = DES.bin_to_dec(
                    int(xor_x[j * 6 + 1] + xor_x[j * 6 + 2] + xor_x[j * 6 + 3] + xor_x[j * 6 + 4]))
                val = sbox[j][row][col]
                sbox_str = sbox_str + DES.dec_to_bin(val)

            # Straight D-box: After substituting rearranging the bits
            sbox_str = DES.permute(sbox_str, per, 32)

            # XOR left and sbox_str
            result = DES.xor(left, sbox_str)
            left = result

            # Swapper
            if i != 15:
                left, right = right, left

        # Combination
        combine = left + right

        # Final permutation: final rearranging of bits to get cipher text
        cipher_text = DES.permute(combine, final_perm, 64)
        return cipher_text


class Decryption(DES):
    def __init__(self, key):
        super().__init__(key)

    @staticmethod
    def decrypt(cipher_text, rkb, rk):
        rkb_rev = rkb[::-1]
        rk_rev = rk[::-1]
        text = DES.bin_to_hex(Encryption.encrypt(cipher_text, rkb_rev, rk_rev))
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
    def generate_sboxes(num_sboxes):
        if DES.sboxes is None:
            DES.sboxes = []
        for _ in range(num_sboxes):
            sbox = np.arange(0, 32)  
            np.random.shuffle(sbox)
            if len(sbox) < 64:
                repetitions = (64 - len(sbox)) // len(sbox) + 1
                sbox = np.tile(sbox, repetitions)[:64]
            sbox = np.reshape(sbox, (4, 16))
            # Round the values to the nearest integer
            sbox = np.round(sbox).astype(int)
            DES.sboxes.append(sbox)
        return DES.sboxes
   

    @staticmethod
    def sbox_array(input_array):
        sbox = DES.get_tables()[3][0]  
        return np.vectorize(sbox)(input_array)

    def differential_table(sbox):
        table = {}
        for input_diff in range(16):
            for output_diff in range(16):
                count = 0
                for input_val in range(16):
                    if sbox[input_val] ^ sbox[input_val ^ input_diff] == output_diff:
                        count += 1
                table[(input_diff, output_diff)] = count
        return table

    def linear_table(sbox):
        table = {}
        for input_mask in range(16):
            for output_mask in range(16):
                count = 0
                for input_val in range(16):
                    if bin(input_val & input_mask).count('1') ^ bin(sbox[input_val] & output_mask).count('1') == 0:
                        count += 1
                table[(input_mask, output_mask)] = count
        return table

    def print_table(table):
        for k, v in table.items():
            print(k, ':', v)

    @staticmethod
    def geneticModel(num_sboxes, population_size):
        num_variables = num_sboxes * 64
        lower_bound = 0
        upper_bound = 31  # Upper bound set to the maximum value in the sbox
    
        xl = np.full(num_variables, lower_bound)  # Array filled with lower_bound
        xu = np.full(num_variables, upper_bound)  # Array filled with upper_bound
        problem = SboxProblem(num_sboxes, xl=xl, xu=xu)
        algorithm = GA(pop_size=population_size)
    
        res = minimize(problem, algorithm, seed=1, verbose=False)
    
        best_solution = np.round(res.X).astype(int)  # Round the solution to integers
        best_score = res.F[0]
    
        return best_solution, best_score

    @staticmethod
    def calculate_dlct_score(sboxes):
        dlct_score_total = 0
        for sbox in sboxes:
            dlct_score_sbox = 0
            n = len(sbox)
            sbox = sbox.astype(int)
            for alpha in range(n):
                for beta in range(n):
                    count_diff = np.sum(np.bitwise_xor(
                        sbox, np.roll(sbox, alpha, axis=0)) == beta)
                    dlct_score_sbox += abs(count_diff - (n / 2))
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
    pt = "0123456789ABCDEF"
    key = "133457799BBCDFF1"
    print(" Before the encription Plain Text : ", pt)

    # Key generation
    key = DES.hex_to_bin(key)
    key = DES.permute(key, keyp, 56)

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

    left = key[0:28]
    right = key[28:56]

    rkb = []
    rk = []
    for i in range(0, 16):
        left = DES.shift_left(left, shift_table[i])
        right = DES.shift_left(right, shift_table[i])

        combine_str = left + right
        round_key = DES.permute(combine_str, key_comp, 48)

        rkb.append(round_key)
        rk.append(DES.bin_to_hex(round_key))

    print("Encryption")
    cipher_text = DES.bin_to_hex(Encryption.encrypt(pt, rkb, rk))
    print("Cipher Text : ", cipher_text)

    print("Decryption")
    plain_text = Decryption.decrypt(cipher_text, rkb, rk)
    print("Plain Text : ", plain_text)

    num_sboxes = 8
    population_size = 100
    best_sboxes, best_score = AIMODEL.geneticModel(num_sboxes, population_size)
    print("Mejor S-box encontrada:")
    print(best_sboxes.reshape(num_sboxes, 4, 16))
    print("Puntaje DLCT de la mejor S-box:", best_score)

    sbox = DES.get_tables()[3][0].flatten()
    print("Differential Table:")
    diff_table = AIMODEL.differential_table(sbox)

    table_data = [(input_diff, output_diff, count)
                  for (input_diff, output_diff), count in diff_table.items()]

    print(tabulate(table_data, headers=[
          "Input Diff", "Output Diff", "Count"], tablefmt="grid"))
    # AIMODEL.print_table(diff_table)

    print("\nLinear Table:")
    lin_table = AIMODEL.linear_table(sbox)
    AIMODEL.print_table(lin_table)


if __name__ == "__main__":
    main()
