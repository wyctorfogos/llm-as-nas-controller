import pandas as pd
import os
import re

def preprocess_patient_data(row, column_names):
    """
    Processa os dados do paciente e retorna um dicionário com os valores filtrados.
    """
    filtered_data = {col: str(row[col]) for col in column_names if pd.notna(row[col]) and row[col] != ""}
    return filtered_data
  
def load_dataset(file_path: str):
    try:
        df = pd.read_csv(file_path)
        return df.columns.tolist(), df
    except Exception as e:
        print(f"Erro ao ler o arquivo. Erro: {e}")
        return None, None

def mounting_prompt(data):
    try:
        # Monta o prompt utilizando os valores do dicionário
        prompt = f'''
            - Patient ID: {data.get('patient_id', 'N/A')}
            - Lesion ID: {data.get('lesion_id', 'N/A')}
            - Age: {data.get('age', 'N/A')} years old
            - Gender: {data.get('gender', 'N/A')}
            - Lesion Location: {data.get('region', 'N/A')}
            - Lesion Size: {data.get('diameter_1', 'N/A')} x {data.get('diameter_2', 'N/A')} mm
            - Fitspatrick: {data.get('fitspatrick', 'N/A')}            
            - Family Medical History:
                - Father: {data.get('background_father', 'N/A')}
                - Mother: {data.get('background_mother', 'N/A')}
            - Environmental Factors:
                - Has Piped Water: {data.get('has_piped_water', 'N/A')}
                - Has Sewage System: {data.get('has_sewage_system', 'N/A')}
                - Pesticide Exposure: {data.get('pesticide', 'N/A')}
            - Medical History:
                - Skin Cancer History: {data.get('skin_cancer_history', 'N/A')}
                - Family Cancer History: {data.get('cancer_history', 'N/A')}
            - Lifestyle:
                - Smoker: {data.get('smoke', 'N/A')}
                - Alcohol Consumption: {data.get('drink', 'N/A')}
            - Symptoms:
                - Itching: {data.get('itch', 'N/A')}
                - Growth: {data.get('grew', 'N/A')}
                - Pain: {data.get('hurt', 'N/A')}
                - Changes in Lesion: {data.get('changed', 'N/A')}
                - Bleeding: {data.get('bleed', 'N/A')}
                - Elevation: {data.get('elevation', 'N/A')}'''
        return prompt
    except Exception as e:
        print(f"Erro ao processar os dados. Erro:{e}\n")
        return None
  
def write_dataset_with_sentences(file_folder_path, dataframe):
    """Salva o dataframe contendo os prompts processados."""
    file_path = os.path.join(file_folder_path, "metadata_with_sentences.csv")
    dataframe.to_csv(file_path, index=False, encoding="utf-8", quotechar='"', sep=",")

if __name__ == "__main__":
    file_folder_path = "/home/wytcor/PROJECTs/mestrado-ufes/lab-life/multimodal-skin-lesion-classifier/PAD-UFES-20"
    
    # Carregar dataset original
    columns_names, file_content = load_dataset(os.path.join(file_folder_path, "metadata.csv"))

    if file_content is not None:
        processed_data = []  # Lista para armazenar os dados antes de criar o dataframe

        for _, row in file_content.iterrows():
            data = preprocess_patient_data(row, columns_names)
            query = mounting_prompt(data)



            query_single_line = re.sub(r'\s+', ' ', query.strip())

            # Adiciona os dados processados à lista
            processed_data.append({
                "patient_id" : data.get("patient_id"),
                "img_id" : data.get("img_id"),
                "sentence": query,
                "diagnostic": data.get("diagnostic")
            })

        # Criar DataFrame final e salvar
        dataframe = pd.DataFrame(processed_data)
        write_dataset_with_sentences(file_folder_path, dataframe)
        
        print(f"Arquivo salvo em: {file_folder_path}/metadata_with_sentences.csv")
