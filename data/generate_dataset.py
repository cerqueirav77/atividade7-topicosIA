"""
Passo 1: Geração de Dataset Sintético
Domínio: Programação em Python
Gera 50+ pares (prompt, response) e salva em train.jsonl e test.jsonl
"""

import os
import json
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Tópicos do domínio escolhido (Programação em Python)
TOPICS = [
    "listas e list comprehensions",
    "dicionários e manipulação de dicionários",
    "funções lambda e funções de ordem superior",
    "tratamento de exceções com try/except",
    "programação orientada a objetos com classes",
    "decorators em Python",
    "generators e iterators",
    "manipulação de arquivos com open()",
    "módulo os e pathlib",
    "expressões regulares com re",
]


def generate_pairs(topic: str, n: int = 6) -> list[dict]:
    """Gera n pares (prompt, response) para um tópico via GPT."""
    system_prompt = (
        "Você é um professor de Python. Gere exatamente {n} pares de pergunta e resposta "
        "sobre o tópico informado. Responda APENAS com um JSON válido, sem nenhum texto antes "
        "ou depois, no seguinte formato:\n"
        '[{{"prompt": "pergunta aqui", "response": "resposta detalhada aqui"}}, ...]'
    ).format(n=n)

    user_prompt = f"Tópico: {topic}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content.strip()

    # Remove possíveis blocos de markdown (```json ... ```)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    pairs = json.loads(content)
    return pairs


def main():
    all_pairs = []

    print(f"Gerando dataset sintético para {len(TOPICS)} tópicos...\n")

    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i}/{len(TOPICS)}] Gerando pares para: {topic}")
        try:
            pairs = generate_pairs(topic, n=6)
            all_pairs.extend(pairs)
            print(f"  ✓ {len(pairs)} pares gerados (total: {len(all_pairs)})")
        except Exception as e:
            print(f"  ✗ Erro no tópico '{topic}': {e}")

    print(f"\nTotal de pares gerados: {len(all_pairs)}")
    assert len(all_pairs) >= 50, "Dataset deve ter pelo menos 50 pares!"

    # Embaralha e divide: 90% treino, 10% teste
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.9)
    train_data = all_pairs[:split]
    test_data = all_pairs[split:]

    # Salva em .jsonl
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(script_dir, "train.jsonl")
    test_path = os.path.join(script_dir, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDataset salvo com sucesso!")
    print(f"  Treino : {len(train_data)} pares → {train_path}")
    print(f"  Teste  : {len(test_data)} pares  → {test_path}")


if __name__ == "__main__":
    main()