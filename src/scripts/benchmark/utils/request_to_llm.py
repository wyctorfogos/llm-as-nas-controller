import requests
import logging
import re

def request_to_ollama(
    prompt: str,
    model_name: str = "qwen3:0.6b",
    host: str = "http://localhost:11434",
    thinking: bool = False,
    timeout: int = 120,
    **kwargs
):
    url = f"{host}/api/generate"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
    }

    # Modelos com suporte a JSON estruturado e thinking
    supports_json = model_name.startswith(("qwen", "gpt-oss"))
    supports_think = model_name.startswith(("qwen", "gpt-oss"))

    if supports_json:
        payload["format"] = "json"

    if supports_think:
        payload["think"] = thinking

    # Parâmetros extras opcionais (temperature, top_p, etc.)
    for k, v in kwargs.items():
        if k not in payload:
            payload[k] = v

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            logging.error(f"Ollama error ({model_name}): {data['error']}")
            return None

        if "response" not in data:
            logging.error(
                f"Resposta inválida do modelo {model_name}: {data}"
            )
            return None

        return data["response"].strip()

    except requests.exceptions.Timeout:
        logging.error(f"Timeout ao consultar o modelo {model_name}")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Erro HTTP ao consultar {model_name}: {e}")
        return None

    except Exception as e:
        logging.error(f"Erro inesperado ({model_name}): {e}")
        return None

def filter_generated_response(generated_sentence: str) -> str:
    """
    Extrai o PRIMEIRO objeto JSON válido da resposta do LLM.
    Compatível com <think>, texto extra e múltiplos blocos.
    """

    if not generated_sentence:
        logging.info("Resposta vazia do LLM.")
        return None

    # Remove bloco <think> se existir
    if "</think>" in generated_sentence:
        generated_sentence = generated_sentence.split("</think>", 1)[1]

    # Regex para capturar o primeiro JSON {...}
    json_match = re.search(r"\{[\s\S]*\}", generated_sentence)

    if not json_match:
        logging.info(
            "Nenhum objeto JSON encontrado na resposta do LLM."
        )
        return None


    return json_match.group(0).strip()
