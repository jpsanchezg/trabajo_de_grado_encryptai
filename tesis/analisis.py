
import numpy as np

# esto es para el uso de todo el procesador para el cálculo de la correlación
import concurrent.futures
from tqdm import tqdm
from functools import partial


import matplotlib.pyplot as plt
import seaborn as sns
import cryptanalysis  


from collections import defaultdict

class Test:
    

    @staticmethod
    def linear_table(sboxes):
        table = {}
        for input_mask in range(16):
            for output_mask in range(16):
                count = 0
                for input_val in range(16):
                    sbox_row = sboxes[input_val]
                    for sbox_val in np.nditer(sbox_row):
                        if bin(input_val & input_mask).count('1') ^ bin(sbox_val & output_mask).count('1') == 0:
                            count += 1
                table[(input_mask, output_mask)] = count
        return table

    @staticmethod
    def compute_correlation(input_mask, output_mask, sbox):
        correlation = 0
        sum_ = 0
        n = int(np.log2(len(sbox)))
        input_mask = int(input_mask)
        output_mask = int(output_mask)

        for input_val in range(2 ** n):
            # XOR del valor de entrada con la máscara de entrada
            input_val_xor_input_mask = input_val ^ (input_val & input_mask)
            output_val = sbox[input_val]

            if isinstance(output_val, np.ndarray):
                output_val_scalar = np.sum(output_val)
            else:
                output_val_scalar = output_val

            if isinstance(output_val_scalar, np.ndarray):
                output_val_scalar = output_val_scalar.item()

            output_val_scalar = int(output_val_scalar)
            output_val_xor_output_mask = output_val_scalar ^ (
                output_val_scalar & output_mask)
            sum_ += bin(input_val_xor_input_mask).count(
                '1') ^ bin(output_val_xor_output_mask).count('1')

        correlation = (-1) ** bin(sum_).count('1')
        correlation = abs(correlation / (2 ** n))
        return correlation

    @staticmethod
    def linearity(sbox):
        n = int(np.log2(len(sbox)))
        max_abs_correlation = 0
        total_iterations = (2 ** n - 1) ** 2
        chunk_size = 4000

        chunks = [(input_mask, output_mask) for input_mask in range(
            1, 2 ** n) for output_mask in range(1, 2 ** n)]

        correlations = []

        with concurrent.futures.ProcessPoolExecutor() as executor, tqdm(total=total_iterations) as pbar:
            for i in range(0, len(chunks), chunk_size):
                chunk = chunks[i:i + chunk_size]
                compute_partial = partial(
                    Test.compute_correlation, sbox=sbox)
                futures = [executor.submit(compute_partial, *params)
                           for params in chunk]
                for future in concurrent.futures.as_completed(futures):
                    correlation = future.result()
                    correlations.append(correlation)
                    max_abs_correlation = max(max_abs_correlation, correlation)
                    pbar.update(len(chunk))

        linearity_value = 1 - 2 * max_abs_correlation

        return correlations, linearity_value

    @staticmethod
    def plot_correlations(correlations, title):
        plt.figure(figsize=(10, 6))
        plt.hist(correlations, bins=50, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel('Correlation')
        plt.ylabel('Frequency')
        plt.show()

    def plot_table(table, title):
        fig, ax = plt.subplots()
        ax.bar(range(len(table)), list(table.values()), align='center')
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table.keys(), rotation=90)
        ax.set_xlabel('Diferencia de entrada/salida')
        ax.set_ylabel('Recuento')
        ax.set_title(title)
        plt.show()
    
    def calculate_average_sac(sbox_row):
        sbox = sbox_row.reshape(1, -1)

        sac_values = []
        for i in range(8):  # Para cada bit de entrada
            changed_bits = []
            # Para cada valor de entrada en la fila (usando sbox.shape[1])
            for x in range(sbox.shape[1]):
                x_flipped = x ^ (1 << i)  # Cambiar el bit i
                # Obtener el elemento de la fila (ahora en la matriz 2D)
                y = sbox[0, x]
                # Obtener el elemento de la fila con el bit cambiado (usando %)
                y_flipped = sbox[0, x_flipped % sbox.shape[1]]
                # Contar bits diferentes
                changed_bits.append(bin(y ^ y_flipped).count("1"))
            # Promedio de bits cambiados (normalizado)
            sac_values.append(sum(changed_bits) / 512)
    
        return np.mean(sac_values)
