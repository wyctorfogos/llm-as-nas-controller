import csv
import os

def save_experiment_log(log_file, params, metrics):
    # Verifica se o arquivo existe; se não, cria e escreve o cabeçalho
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as csvfile:
        fieldnames = list(params.keys()) + list(metrics.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Combina dicionários de parâmetros e métricas para escrever uma linha
        row_data = {**params, **metrics}
        writer.writerow(row_data)
