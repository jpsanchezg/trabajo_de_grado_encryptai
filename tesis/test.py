
from Aimodel import AI_MODEL
import numpy as np

# esto es para el uso de todo el procesador para el cálculo de la correlación
import concurrent.futures
from tqdm import tqdm
from functools import partial
class Test:
    @staticmethod
    def differential_table(sbox):
        table = {}
        for input_diff in range(16):
            for output_diff in range(16):
                xor_func = np.vectorize(lambda x: x ^ input_diff)
                count = np.sum(sbox ^ xor_func(sbox) == output_diff)
                table[(input_diff, output_diff)] = count
        return table

    @staticmethod
    def linear_table(sboxes):
        table = {}
        for input_mask in range(16):
            for output_mask in range(16):
                count = 0
                for input_val in range(16):
                    sbox_row = sboxes[input_val]
                    # Use np.nditer to iterate over elements of the numpy array
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
        input_mask = int(input_mask)  # Convertir a entero si no lo es
        output_mask = int(output_mask)  # Convertir a entero si no lo es
        for input_val in range(0, 2 ** n):
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
        correlation += (-1) ** bin(sum_).count('1')
        correlation = abs(correlation / (2 ** n))
        return correlation

    @staticmethod
    def linearity(sbox):
        n = int(np.log2(len(sbox)))
        max_abs_correlation = 0

        total_iterations = (2 ** n - 1) ** 2  # Total de iteraciones
        chunk_size = 4000

        chunks = []
        for input_mask in range(1, 2 ** n):
            for output_mask in range(1, 2 ** n):
                chunks.append((input_mask, output_mask))

        with concurrent.futures.ProcessPoolExecutor() as executor, tqdm(total=total_iterations) as pbar:
            for i in range(0, len(chunks), chunk_size):
                chunk = chunks[i:i+chunk_size]
                compute_partial = partial(
                    Test.compute_correlation, sbox=sbox)
                futures = [executor.submit(compute_partial, *params)
                           for params in chunk]
                for future in concurrent.futures.as_completed(futures):
                    correlation = future.result()
                    max_abs_correlation = max(max_abs_correlation, correlation)
                    pbar.update(len(chunk))

        return 1 - 2 * max_abs_correlation

    def plot_table(table, title):
        fig, ax = plt.subplots()
        ax.bar(range(len(table)), list(table.values()), align='center')
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table.keys(), rotation=90)
        ax.set_xlabel('Diferencia de entrada/salida')
        ax.set_ylabel('Recuento')
        ax.set_title(title)
        plt.show()
