import time
import matplotlib.pyplot as plt
import numpy as np
from des import DES, Encryption, Decryption, AIMODEL

def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

def compare_des_performance(pt, key_without_AI, key_with_AI, rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI, sboxes_with_AI):
    # Measure time for encryption and decryption without AI
    cipher_text_without_AI, enc_time_without_AI = measure_time(Encryption.encrypt, pt, rkb_without_AI, rk_without_AI, DES.sboxes)
    plain_text_without_AI, dec_time_without_AI = measure_time(Decryption.decrypt, cipher_text_without_AI, rkb_without_AI, rk_without_AI, DES.sboxes)
    
    # Measure time for encryption and decryption with AI
    cipher_text_with_AI, enc_time_with_AI = measure_time(Encryption.encrypt, pt, rkb_with_AI, rk_with_AI, sboxes_with_AI)
    plain_text_with_AI, dec_time_with_AI = measure_time(Decryption.decrypt, cipher_text_with_AI, rkb_with_AI, rk_with_AI, sboxes_with_AI)
    
    # Display metrics
    print("Encryption Time without AI:", enc_time_without_AI)
    print("Decryption Time without AI:", dec_time_without_AI)
    print("Encryption Time with AI:", enc_time_with_AI)
    print("Decryption Time with AI:", dec_time_with_AI)
    
    # Plot results
    labels = ['Encryption without AI', 'Decryption without AI', 'Encryption with AI', 'Decryption with AI']
    times = [enc_time_without_AI, dec_time_without_AI, enc_time_with_AI, dec_time_with_AI]

    plt.bar(labels, times)
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of DES Performance')
    plt.show()

def brute_force_des(pt, cipher_text, sboxes, SHIFT_TABLE, KEY_COMP):
    for possible_key in range(2**48):  # Probamos todas las claves posibles
        possible_key_str = format(possible_key, '048b')  # Convertimos la clave en binario
        key = DES.bin_to_hex(possible_key_str)  # Convertimos la clave a formato hexadecimal

        # Generamos las rondas de claves
        rkb = []
        rk = []
        left = possible_key_str[:28]
        right = possible_key_str[28:56]
        for i in range(0, 16):
            left = DES.shift_left(left, SHIFT_TABLE[i])
            right = DES.shift_left(right, SHIFT_TABLE[i])

            combine_str = left + right
            round_key = DES.permute(combine_str, KEY_COMP, 48)

            rkb.append(round_key)
            rk.append(DES.bin_to_hex(round_key))

        # Probamos la clave generada
        decrypted_text = Decryption.decrypt(cipher_text, rkb, rk, sboxes)
        if decrypted_text == pt:
            return key  # Si el texto descifrado coincide con el texto plano, hemos encontrado la clave
    return None  # Si no se encuentra ninguna clave, retornamos None

def brute_force_des_with_ai(pt, cipher_text, best_sboxes, SHIFT_TABLE, KEY_COMP):
    for possible_key in range(2**48):  # Probamos todas las claves posibles
        possible_key_str = format(possible_key, '048b')  # Convertimos la clave en binario
        key = DES.bin_to_hex(possible_key_str)  # Convertimos la clave a formato hexadecimal

        # Generamos las rondas de claves
        rkb = []
        rk = []
        left = possible_key_str[:28]
        right = possible_key_str[28:56]
        for i in range(0, 16):
            left = DES.shift_left(left, SHIFT_TABLE[i])
            right = DES.shift_left(right, SHIFT_TABLE[i])

            combine_str = left + right
            round_key = DES.permute(combine_str, KEY_COMP, 48)

            rkb.append(round_key)
            rk.append(DES.bin_to_hex(round_key))

        # Probamos la clave generada
        decrypted_text = Decryption.decrypt(cipher_text, rkb, rk, best_sboxes)
        if decrypted_text == pt:
            return key  # Si el texto descifrado coincide con el texto plano, hemos encontrado la clave
    return None  # Si no se encuentra ninguna clave, retornamos None


def main():
    pt = "ADC0326456789BAEF"
    print(" Before the encryption Plain Text : ", pt)

    # Generate S-boxes
    sboxes_with_AI = AIMODEL.generate_sboxes(8)

    keyp = [
    57, 49, 41, 33, 25, 17, 9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 63, 55, 47, 39,
    31, 23, 15, 7, 62, 54, 46, 38,
    30, 22, 14, 6, 61, 53, 45, 37,
    29, 21, 13, 5, 28, 20, 12, 4
    ]

    # Key generation for DES algorithm without AI
    key_without_AI = DES.generate_key()
    key_without_AI = DES.hex_to_bin(key_without_AI)
    key_without_AI = DES.permute(key_without_AI, keyp, 56)

    # Key generation for DES algorithm with AI
    key_with_AI = DES.generate_key()
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

    # DES with AI
    num_sboxes = 8
    population_size = 100
    best_sboxes, _ = AIMODEL.geneticModel(num_sboxes, population_size)
    sboxes_with_AI = best_sboxes.reshape(num_sboxes, 4, 16)

    # Compare DES performance
    compare_des_performance(pt, key_without_AI, key_with_AI, rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI, sboxes_with_AI)

    # Cifrado del texto plano con DES sin IA
    cipher_text_without_AI = Encryption.encrypt(pt, rkb_without_AI, rk_without_AI, DES.sboxes)

    # Cifrado del texto plano con DES con IA
    cipher_text_with_AI = Encryption.encrypt(pt, rkb_with_AI, rk_with_AI, sboxes_with_AI)

    # Ataque de fuerza bruta al DES tradicional
    key_found = brute_force_des(pt, cipher_text_without_AI, DES.sboxes, shift_table, key_comp)
    print("Key found by brute force DES traditional:", key_found)

    # Ataque de fuerza bruta al DES con IA
    key_found_with_ai = brute_force_des_with_ai(pt, cipher_text_with_AI, sboxes_with_AI, shift_table, key_comp)
    print("Key found by brute force DES with AI:", key_found_with_ai)


if __name__ == "__main__":
    main()