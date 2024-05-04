from des import DES
from Encryption import Encryption, Decryption
from Aimodel import AIMODEL

keyp = [
    57, 49, 41, 33, 25, 17, 9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 63, 55, 47, 39,
    31, 23, 15, 7, 62, 54, 46, 38,
    30, 22, 14, 6, 61, 53, 45, 37,
    29, 21, 13, 5, 28, 20, 12, 4
]


def main():
    pt = "ADC0326456789BAEF"
    print(" Before the encription Plain Text : ", pt)
    num_sboxes = 32
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
