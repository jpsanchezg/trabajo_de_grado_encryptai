from des import DES
from Encryption import Encryption, Decryption
from Aimodel import AI_MODEL, QLearningAgent
import numpy as np
import random
import binascii

from analisis import Test



def limitar_y_mapear(cadena_hex, diccionario):
    resultado = ""
    for caracter in cadena_hex[:16]:
        # Mapear usando el diccionario, o mantener el carácter original si no está en el diccionario
        resultado += diccionario.get(caracter, caracter)
    return resultado

def main():
    diccionario_hex = {
        '0': 'A', '1': 'B', '2': 'C', '3': 'D',
        '4': 'E', '5': 'F', '6': '0', '7': '1',
        '8': '2', '9': '3', 'A': '4', 'B': '5',
        'C': '6', 'D': '7', 'E': '8', 'F': '9'
    }
    mensaje_original = input("Ingresa el mensaje que deseas codificar: ")

    mensaje_bytes = mensaje_original.encode('utf-8')
    
    diccionario_hex_inverso = {
        valor: clave for clave, valor in diccionario_hex.items()}

    mensaje_hex = binascii.hexlify(mensaje_bytes).decode('utf-8')
    resultado_final = limitar_y_mapear(mensaje_hex, diccionario_hex)
    resultado_final = resultado_final.upper()
    pt = resultado_final
    print(" Before the encription Plain Text : ", pt)
    keyp = [
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 27, 19, 11, 3,
        60, 52, 44, 36, 63, 55, 47, 39,
        31, 23, 15, 7, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 28, 20, 12, 4
    ]

    des_sbox = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]],

                [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]],

                [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]],

                [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]],

                [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]],

                [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]],

                [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]],

                [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]]]
    num_sboxes = 16
    print("num_sboxes: ", num_sboxes)
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

    print("Encryption without AI")
    cipher_text_without_AI = DES.bin_to_hex(
        Encryption.encrypt(pt, rkb_without_AI, rk_without_AI, des_sbox))
    print("Cipher Text without AI: ", cipher_text_without_AI)

    print("Decryption without AI")
    plain_text_without_AI = Decryption.decrypt(
        cipher_text_without_AI, rkb_without_AI, rk_without_AI, des_sbox)
    print("Plain Text without AI: ", plain_text_without_AI)
    mensaje_decodificado_hex = limitar_y_mapear(
        plain_text_without_AI, diccionario_hex_inverso)
    # DES with AI

    print("Encryption with Genetic Algorithm with SA ")
    de_sboxes, sa_sboxes = AI_MODEL.generate_sboxes_combined(
        num_sboxes)

    desboxes = np.array(de_sboxes, dtype=np.int32)
    sasboxes = np.array(sa_sboxes, dtype=np.int32)
    #print("Sboxes SA: ", sasboxes)
    #lineality_de = Test.linearity(desboxes)
    #lineality_sa = Test.linearity(sasboxes)
    #print("Lineal correlation Sboxes Des:", lineality_de)
    #print("Lineal correlation Sboxes SA:", lineality_sa)


    # Use combined sboxes in geneticModel:
    sbox_genetic, best_score = AI_MODEL.geneticModel_combined(num_sboxes)

    best_score = np.round(best_score, 2)
    print("Encryption with AI")
    cipher_text_with_AI = DES.bin_to_hex(Encryption.encrypt(
        pt, rkb_with_AI, rk_with_AI, sbox_genetic.reshape(num_sboxes, 4, 16)))
    
    print("Cipher Text with AI: ", cipher_text_with_AI)

    print("Decryption with AI")
    plain_text_with_AI = Decryption.decrypt(
        cipher_text_with_AI, rkb_with_AI, rk_with_AI, sbox_genetic.reshape(num_sboxes, 4, 16))
    print("Decrypt result text with AI: ", plain_text_with_AI)
    print("Best Solution:", sbox_genetic)

    print("Encryption with Q-Learning")
    agent = QLearningAgent(num_sboxes)
    sbox_qlearning, best_score = agent.train()
    print("Best Solution:", sbox_qlearning)
    print("Best Score:", best_score)

    print("Encryption with AI")
    cipher_text_with_AI = DES.bin_to_hex(Encryption.encrypt(
        pt, rkb_with_AI, rk_with_AI, sbox_qlearning.reshape(num_sboxes, 4, 16)))
    
    print("Cipher Text with AI: ", cipher_text_with_AI)

    print("Decryption with AI")
    plain_text_with_AI = Decryption.decrypt(
        cipher_text_with_AI, rkb_with_AI, rk_with_AI, sbox_qlearning.reshape(num_sboxes, 4, 16))
    print("Decrypt result text with AI: ", plain_text_with_AI)
    print("")
    print("")
    #print("Testing the Nonlinearity of the S-boxes")
    #print("")
    #Test the linearity of the S-boxes
    #sboxdes = np.array(des_sbox, dtype=np.int32)
    #linearity_values_des = Test.linearity(sboxdes)
    #print("")
    #
#
    sboxgenetic = np.array(sbox_genetic, dtype=np.int32)
    linearity_values_genetic = Test.linearity(sboxgenetic)
    print("")
    
#
    sboxqlearning = np.array(sbox_qlearning, dtype=np.int32)
    linearity_values_qlearning = Test.linearity(sboxqlearning)
    print("")
    print("NonLineal correlation Sboxes Des:", linearity_values_des)
    print("")
    print("NonLineal correlation Sboxes Genetic:", linearity_values_genetic)
    print("")
    print("NonLineal correlation Sboxes Qlearning:", linearity_values_qlearning)


    # preparando las sboxes para el SAC (SAC: Strict Avalanche Criterion)


    
    print("")
    print("")
    print("Genetic Algorithm")
    print("")

    sbox_genetic =  [sbox_genetic[i:i+16] for i in range(0, 256, 16)]
    print("Sboxes Genetic: ", sbox_genetic)
    genetic_sac = Test.calculate_average_sac_for_multiple_sboxes(sbox_genetic)
    print(f"Average SAC for Genetic sboxes= {genetic_sac:.4f}")

    print("")
    print("")
    print("Q Learning")
    print("")
    sbox_qlearning =  [sbox_qlearning[i:i+16] for i in range(0, 256, 16)]
    qlearning_sac = Test.calculate_average_sac_for_multiple_sboxes(sbox_qlearning)
    print(f"Average SAC for Q Learning sboxes= {qlearning_sac:.4f}")
    print("")
    print("")


if __name__ == "__main__":
    main()
