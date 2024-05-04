
import secrets




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
        num = int(num)
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
