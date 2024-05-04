
from des import DES


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
