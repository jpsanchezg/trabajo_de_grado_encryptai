import time
import numpy as np
from des import DES, Encryption, AIMODEL

def generate_plaintext_pairs(num_pairs):
    """
    Genera pares de textos claros aleatorios de 16 bits.

    Args:
    - num_pairs: Número de pares de textos claros a generar.

    Returns:
    - Lista de pares de textos claros.
    """
    plaintext_pairs = []
    for _ in range(num_pairs):
        # Genera dos textos claros aleatorios de 16 bits
        plaintext1 = ''.join([np.random.choice(['0', '1']) for _ in range(16)])
        plaintext2 = ''.join([np.random.choice(['0', '1']) for _ in range(16)])
        # Agrega el par de textos claros a la lista
        plaintext_pairs.append((plaintext1, plaintext2))
    return plaintext_pairs

def encrypt_with_des(plaintext, round_keys, sboxes):
    """
    Cifra un texto claro utilizando el algoritmo de cifrado DES con las S-boxes especificadas.

    Args:
    - plaintext: Texto claro a cifrar.
    - round_keys: Claves de ronda generadas por DES.
    - sboxes: S-boxes utilizadas en el cifrado DES.

    Returns:
    - Cifrado del texto claro.
    """
    # Cifra el texto claro utilizando el algoritmo DES
    ciphertext = Encryption.encrypt(plaintext, round_keys, sboxes)
    return ciphertext

def differential_attack_on_aimodel(plaintext_pairs, round_keys_original):
    """
    Realiza un ataque diferencial utilizando las S-boxes generadas por AIMODEL.

    Args:
    - plaintext_pairs: Pares de textos claros.
    - round_keys_original: Claves de ronda generadas por DES.

    Returns:
    - Tasa de éxito del ataque.
    - Tiempo de ejecución del ataque.
    """
    start_time = time.time()  # Registra el tiempo de inicio del ataque
    successful_attacks = 0  # Contador de ataques exitosos
    num_pairs = len(plaintext_pairs)  # Número total de pares de textos claros
    # Genera las mejores S-boxes utilizando AIMODEL
    num_sboxes = 8
    population_size = 100
    best_sboxes, _ = AIMODEL.geneticModel(num_sboxes, population_size)
    # Convierte las S-boxes en un formato compatible con DES
    sboxes = best_sboxes.reshape(num_sboxes, 4, 16)
    # Realiza el ataque diferencial
    for plaintext1, plaintext2 in plaintext_pairs:
        # Cifra los textos claros utilizando las S-boxes originales de DES y las S-boxes generadas por AIMODEL
        ciphertext_original = encrypt_with_des(plaintext1, round_keys_original, sboxes)
        ciphertext_aimodel = encrypt_with_des(plaintext2, round_keys_original, sboxes)
        # Verifica si los cifrados son iguales, lo que indica un posible éxito del ataque
        if ciphertext_original == ciphertext_aimodel:
            successful_attacks += 1  # Incrementa el contador de ataques exitosos
    end_time = time.time()  # Registra el tiempo de finalización del ataque
    execution_time = end_time - start_time  # Calcula el tiempo total de ejecución del ataque
    # Calcula la tasa de éxito del ataque (porcentaje de ataques exitosos)
    success_rate = successful_attacks / num_pairs
    return success_rate, execution_time

def calculate_dlct_score(sboxes):
    """
    Calcula el puntaje DLCT (Differential Linearity and Correlation Table) de las S-boxes.

    Args:
    - sboxes: S-boxes para las cuales se calculará el puntaje DLCT.

    Returns:
    - Puntaje DLCT total de las S-boxes.
    """
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

def main():
    num_pairs = 100
    plaintext_pairs = generate_plaintext_pairs(num_pairs)  # Genera pares de textos claros aleatorios
    print("Ataque diferencial al cifrado DES con AIMODEL:")
    key = "133457799BBCDFF1"  # Clave utilizada en el cifrado DES
    key = DES.hex_to_bin(key)
    keyp = [
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 27, 19, 11, 3,
        60, 52, 44, 36, 63, 55, 47, 39,
        31, 23, 15, 7, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 28, 20, 12, 4
    ]
    des_instance = DES(key)  # Crea una instancia de la clase DES
    round_keys_original = des_instance.generate_round_keys(des_instance.permute(key, keyp, 56))  # Genera claves de ronda
    # Realiza el ataque diferencial utilizando las S-boxes generadas por AIMODEL
    success_rate, execution_time = differential_attack_on_aimodel(plaintext_pairs, round_keys_original)
    print("Tasa de éxito del ataque diferencial:", success_rate)
    print("Tiempo de ejecución del ataque diferencial:", execution_time, "segundos")

    # Calcula el puntaje DLCT de las S-boxes originales de DES
    sboxes_original = DES.get_tables()[3]  # Obtiene las S-boxes originales de DES
    dlct_score_original = calculate_dlct_score(sboxes_original)  # Calcula el puntaje DLCT
    print("Puntaje DLCT de las S-boxes originales de DES:", dlct_score_original)

    # Calcula el puntaje DLCT de las S-boxes generadas por AIMODEL
    num_sboxes = 8
    population_size = 100
    best_sboxes, _ = AIMODEL.geneticModel(num_sboxes, population_size)  # Genera las mejores S-boxes con AIMODEL
    dlct_score_aimodel = calculate_dlct_score(best_sboxes.reshape(num_sboxes, 4, 16))  # Calcula el puntaje DLCT
    print("Puntaje DLCT de las S-boxes generadas por AIMODEL:", dlct_score_aimodel)

if __name__ == "__main__":
    main()