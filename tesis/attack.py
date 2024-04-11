import time
import matplotlib.pyplot as plt
from des import DES, Encryption, Decryption, AIMODEL
import multiprocessing


def measure_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def compare_des_performance(pt, key_without_AI, key_with_AI, rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI, sboxes_with_AI, sboxes_without_AI):
    # Measure time for encryption and decryption without AI
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results_without_AI = pool.starmap(measure_time, [
        (Encryption.encrypt, pt, rkb_without_AI, rk_without_AI, sboxes_without_AI),
        (Decryption.decrypt, pt, rkb_without_AI,
         rk_without_AI, sboxes_without_AI)
    ])
    pool.close()

    # Measure time for encryption and decryption with AI
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results_with_AI = pool.starmap(measure_time, [
        (Encryption.encrypt, pt, rkb_with_AI, rk_with_AI, sboxes_with_AI),
        (Decryption.decrypt, pt, rkb_with_AI,
         rk_with_AI, sboxes_with_AI)
    ])
    pool.close()

    # Display metrics
    labels = ['Encryption without AI', 'Decryption without AI',
              'Encryption with AI', 'Decryption with AI']
    times = [result[1] for result in results_without_AI + results_with_AI]

    for label, time_taken in zip(labels, times):
        print(f"{label}: {time_taken}")

    # Plot results
    plt.bar(labels, times)
    plt.ylabel('Time (seconds)')
    plt.title('Comparison of DES Performance')
    plt.show()


def brute_force_attack(args):
    cipher_text, shift_table, keyp, key_comp, sboxes, original_plain_text, start, end = args
    print("Atacando desde", start, "hasta", end)
    for possible_key in range(start, end):
        key_binary = format(possible_key, '064b')
        key_permuted = DES.permute(key_binary, keyp, 56)
        left_key = key_permuted[:28]
        right_key = key_permuted[28:]
        rkb, rk = generate_round_keys(
            left_key, right_key, shift_table, key_comp)
        decrypted_text = Decryption.decrypt(cipher_text, rkb, rk, sboxes)
        if decrypted_text == original_plain_text:
            print("Clave encontrada:", DES.bin_to_hex(key_permuted))
            return  # Exit the function immediately after finding the key
    return None


def generate_round_keys(left_key, right_key, shift_table, key_comp):
    rkb = []  # Lista para almacenar las claves redondas ampliadas
    rk = []   # Lista para almacenar las claves redondas permutadas

    for i in range(16):
        # Realizar desplazamiento de bits según la tabla de desplazamiento
        left_key = DES.shift_left(left_key, shift_table[i])
        right_key = DES.shift_left(right_key, shift_table[i])

        # Combinar las mitades izquierda y derecha de la clave
        combined_key = left_key + right_key

        # Permutar la clave combinada según la tabla de compresión de claves
        round_key = DES.permute(combined_key, key_comp, 48)

        # Agregar la clave de la ronda actual a las listas rkb y rk
        rkb.append(combined_key)
        rk.append(round_key)

    return rkb, rk


def main():
    """Main function to demonstrate DES performance comparison."""
    # Original plain text
    original_plain_text = "ADC0326456789BAEF"
    print("Original Plain Text:", original_plain_text)

    # Key generation
    keyp = [57, 49, 41, 33, 25, 17, 9, 1,
            58, 50, 42, 34, 26, 18, 10, 2,
            59, 51, 43, 35, 27, 19, 11, 3,
            60, 52, 44, 36, 63, 55, 47, 39,
            31, 23, 15, 7, 62, 54, 46, 38,
            30, 22, 14, 6, 61, 53, 45, 37,
            29, 21, 13, 5, 28, 20, 12, 4]

    key_without_AI = DES.generate_key()
    print("Key without AI:", key_without_AI)
    key_without_AI = DES.hex_to_bin(key_without_AI)
    key_without_AI = DES.permute(key_without_AI, keyp, 56)

    key_with_AI = DES.generate_key()
    key_with_AI = DES.hex_to_bin(key_with_AI)
    key_with_AI = DES.permute(key_with_AI, keyp, 56)

    shift_table = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
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
    # Generate round keys
    rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI = DES.generate_round_keys(
        shift_table, key_comp, left, right, left_with_AI, right_with_AI)

    # DES with AI
    num_sboxes = 8
    population_size = 100
    best_sboxes, _ = AIMODEL.geneticModel(num_sboxes, population_size)
    sboxes_with_AI = best_sboxes.reshape(num_sboxes, 4, 16)
    sboxes_without_AI = AIMODEL.generate_sboxes(8)

    # Compare DES performance
    compare_des_performance(original_plain_text, key_without_AI, key_with_AI,
                            rkb_without_AI, rk_without_AI, rkb_with_AI, rk_with_AI, sboxes_with_AI, sboxes_without_AI)

    # Encryption
    cipher_text_without_AI = Encryption.encrypt(
        original_plain_text, rkb_without_AI, rk_without_AI, sboxes_without_AI)
    cipher_text_with_AI = Encryption.encrypt(
        original_plain_text, rkb_with_AI, rk_with_AI, sboxes_with_AI)

    print("Cipher Text without AI:", cipher_text_without_AI)
    print("Cipher Text with AI:", cipher_text_with_AI)

    # Attack
    print("Iniciando ataque de fuerza bruta...")
    start_time = time.time()

    num_processes = multiprocessing.cpu_count()  # Number of CPU cores
    chunk_size = 2**20  # Chunk size for each process
    total_keys = 2**64  # Total number of keys

    # Create a pool of processes
    pool = multiprocessing.Pool(num_processes)

    # Perform brute force attack on each chunk
    for start in range(0, total_keys, chunk_size * num_processes):
        end = min(start + chunk_size * num_processes, total_keys)
        pool.map_async(brute_force_attack, [(cipher_text_without_AI, shift_table, keyp, key_comp,
                                             sboxes_without_AI, original_plain_text,
                                             i*chunk_size, (i+1)*chunk_size)
                                            for i in range(start, end, chunk_size)])

    pool.close()
    pool.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Tiempo de ejecución:", elapsed_time, "segundos")


if __name__ == "__main__":
    main()
