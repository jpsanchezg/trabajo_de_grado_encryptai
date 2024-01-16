def hex_to_binary(hex_string):
    return ''.join(format(int(c, 16), '04b') for c in hex_string)

def binary_to_hex(binary_string):
    return hex(int(binary_string, 2))[2:]
def permute(initial, permutation):
    return [initial[i - 1] for i in permutation]

def initial_permutation(data_block):
    permutation = [
        58, 50, 42, 34, 26, 18, 10, 2,
        60, 52, 44, 36, 28, 20, 12, 4,
        62, 54, 46, 38, 30, 22, 14, 6,
        64, 56, 48, 40, 32, 24, 16, 8,
        57, 49, 41, 33, 25, 17, 9, 1,
        59, 51, 43, 35, 27, 19, 11, 3,
        61, 53, 45, 37, 29, 21, 13, 5,
        63, 55, 47, 39, 31, 23, 15, 7
    ]
    return permute(data_block, permutation)

def shift_left(data, shifts):
    return data[shifts:] + data[:shifts]

def xor(str1, str2):
    return ''.join(format(ord(a) ^ ord(b), '08b') for a, b in zip(str1, str2))



straight_permutation = [
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
]
# Define the key_compression permutation
key_compression = [
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4,
    26, 8, 16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
]

def generate_subkeys(key, key_length=64):
    if len(key) != key_length:
        raise ValueError(f"The key length must be {key_length} bits")

    key = key_permutation(key)
    shifts = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]
    subkeys = [key[0:28], key[28:56]]

    for round_num in range(1, 17):
        subkeys[0] = shift_left(subkeys[0], shifts[round_num - 1])
        subkeys[1] = shift_left(subkeys[1], shifts[round_num - 1])
        subkey = subkeys[0] + subkeys[1]
        subkeys.append(permute(subkey, key_compression))

    return subkeys

def final_permutation(data_block):
    # Permutation constants
    permutation = [
        40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25
    ]
    return permute(data_block, permutation)
def substitute(sbox_input, sbox):
    row = int(sbox_input[0] + sbox_input[5], 2)
    col = int(sbox_input[1:5], 2)
    return format(sbox[row][col], '04b')



def expansion_permutation(data_block):
    # Permutation constants
    permutation = [
        32, 1, 2, 3, 4, 5, 4, 5,
        6, 7, 8, 9, 8, 9, 10, 11,
        12, 13, 12, 13, 14, 15, 16, 17,
        16, 17, 18, 19, 20, 21, 20, 21,
        22, 23, 24, 25, 24, 25, 26, 27,
        28, 29, 28, 29, 30, 31, 32, 1
    ]
    return permute(data_block, permutation)

def key_permutation(key):
    # Permutation constants
    permutation = [
        57, 49, 41, 33, 25, 17, 9, 1,
        58, 50, 42, 34, 26, 18, 10, 2,
        59, 51, 43, 35, 27, 19, 11, 3,
        60, 52, 44, 36, 63, 55, 47, 39,
        31, 23, 15, 7, 62, 54, 46, 38,
        30, 22, 14, 6, 61, 53, 45, 37,
        29, 21, 13, 5, 28, 20, 12, 4
    ]
    return permute(key, permutation)

def feistel_network(right_block, subkey):
    expanded_block = expansion_permutation(right_block)
    xor_result = xor(expanded_block, subkey)

    print("xor_result:", xor_result)

    sbox_inputs = [xor_result[i:i+6] for i in range(0, 48, 6)]
    sbox_outputs = [substitute(sbox_input, sbox) for sbox_input, sbox in zip(sbox_inputs, sboxes)]

    sbox_output = ''.join(sbox_outputs)
    permuted_block = permute(sbox_output, straight_permutation)

    return xor(right_block, permuted_block)
sboxes = [[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
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



def des_encrypt_block(data_block, subkeys):
    data_block = initial_permutation(data_block)
    left_block, right_block = data_block[:32], data_block[32:]

    for round_num in range(1, 17):
        left_block, right_block = right_block, xor(left_block, feistel_network(right_block, subkeys[round_num]))

    encrypted_block = final_permutation(right_block + left_block)
    return ''.join(encrypted_block)


def des_encrypt_hex(plaintext_hex, key_hex):
    if len(plaintext_hex) % 16 != 0 or len(key_hex) != 16:
        raise ValueError("Invalid input size. Plaintext should be in multiples of 16 characters, and the key should be 16 characters long.")

    plaintext_binary = hex_to_binary(plaintext_hex)
    key_binary = hex_to_binary(key_hex)

    subkeys = generate_subkeys(key_binary)
    plaintext_blocks = [plaintext_binary[i:i+64] for i in range(0, len(plaintext_binary), 64)]

    encrypted_blocks = [des_encrypt_block(block, subkeys) for block in plaintext_blocks]
    ciphertext_binary = ''.join(encrypted_blocks)

    return binary_to_hex(ciphertext_binary)
def des_encrypt_ai(inputs):
    if len(inputs) != 2:
        raise ValueError("AI should provide a list of two elements: [plaintext_hex, key_hex]")

    plaintext_hex, key_hex = inputs
    return des_encrypt_hex(plaintext_hex, key_hex)
def main():
    inputs = ["123456ABCD132536", "AABB09182736CCDD"]  # Replace with AI-provided inputs
    ciphertext = des_encrypt_ai(inputs)
    print("Original Message (hex):", inputs[0])
    print("Key (hex):", inputs[1])
    print("Encrypted Message (hex):", ciphertext)

if __name__ == "__main__":
    main()