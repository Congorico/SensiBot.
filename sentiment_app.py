import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def carregar_analisador():
    # Carrega um modelo BERT ajustado para análise de sentimentos em português
    nome_modelo = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
    modelo = AutoModelForSequenceClassification.from_pretrained(nome_modelo)
    
    return tokenizer, modelo

def analisar_sentimento(texto, tokenizer, modelo):
    # Transforma o texto em números (tokens) que a rede neural consegue entender
    inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Passa os tokens pela rede neural BERT
    with torch.no_grad():
        outputs = modelo(**inputs)
    
    # O modelo retorna notas de 1 a 5 estrelas
    predicao = torch.argmax(outputs.logits, dim=1).item() + 1
    
    # Converte as estrelas em sentimento legível
    if predicao <= 2:
        return f"🔴 NEGATIVO ({predicao} estrelas) - Alerta de crise!"
    elif predicao == 3:
        return f"🟡 NEUTRO ({predicao} estrelas) - Opinião morna."
    else:
        return f"🟢 POSITIVO ({predicao} estrelas) - Público engajado!"

if __name__ == "__main__":
    print("Inicializando a IA do SentiBot...")
    tok, mod = carregar_analisador()
    print("IA Pronta!\n")
    
    # Simulação de comentários coletados na internet
    comentarios = [
        "Amei o novo design do aplicativo, ficou muito mais rápido de usar!",
        "Esse serviço é horrível. O suporte não responde e o sistema caiu de novo.",
        "Chegou dentro do prazo, mas a caixa veio um pouco amassada."
    ]
    
    for c in comentarios:
        print(f"Texto: '{c}'")
        print(f"Análise: {analisar_sentimento(c, tok, mod)}")
        print("-" * 50)
