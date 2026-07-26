def filter_generated_response(generated_sentence:str=None):
    '''
        Filtra as sentenças geradas por LLM
    '''
    try:
        if "</think>" in generated_sentence:
            after_think = generated_sentence.split("</think>", 1)[1].strip()
            print("✅ Extracted text after </think>:\n")
            # Retorno da sentença filtrada
            return after_think
        else:
            print("❌ No </think> found in text.")
            after_think = generated_sentence
        # Retorno da sentença filtrada
        return after_think

    except Exception as e:
        raise ValueError(f"Erro ao realizar a chamada dos dados:{e}\n")