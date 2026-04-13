# Lab 07 — Fine-tuning de LLMs com LoRA e QLoRA

**Disciplina:** Tópicos em Inteligência Artificial
**Instituição:** Instituto iCEV
**Aluno:** Victor Cerqueira
**Versão:** v1.0

*Partes geradas/complementadas com IA, revisadas por Victor Cerqueira*

---

## O que esse projeto faz

Esse laboratório implementa um pipeline de *fine-tuning* especializado de um LLM usando duas técnicas que tornam o treinamento viável em GPUs comuns:

- **LoRA** — em vez de retreinar o modelo inteiro, treina apenas matrizes pequenas inseridas nas camadas existentes
- **QLoRA** — comprime os pesos do modelo para 4 bits, reduzindo o consumo de memória sem perder muito em qualidade

O modelo escolhido foi o `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, que tem a mesma arquitetura do Llama 2 mas não exige licença e roda tranquilo na GPU T4 gratuita do Google Colab.

---

## Estrutura

```
atividade7-topicosIA/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py   # gera os pares de treino via API da OpenAI
│   ├── train.jsonl           # 54 exemplos de treino
│   └── test.jsonl            # 6 exemplos de teste
└── training/
    └── finetune.py           # pipeline completo: quantização + LoRA + treino
```

---

## Passo a passo para rodar

### Gerando o dataset

Antes de tudo, exporte sua chave da OpenAI:

```bash
export OPENAI_API_KEY="sua-chave-aqui"
pip install openai
python data/generate_dataset.py
```

O script chama a API do GPT-4o-mini para gerar 60 pares de pergunta e resposta sobre Programação em Python, depois divide em 90% treino e 10% teste e salva tudo em `.jsonl`.

### Rodando o fine-tuning (Google Colab)

O treinamento precisa de GPU, então o lugar certo é o Colab com uma T4:

```python
!git clone https://github.com/cerqueirav77/atividade7-topicosIA.git
%cd atividade7-topicosIA
!pip install -q torch transformers datasets peft trl bitsandbytes accelerate scipy
!python training/finetune.py
```

Ao final, o adaptador LoRA é salvo na pasta `./lora_adapter/`.

---

## Detalhes técnicos de cada passo

### Passo 1 — Dataset sintético

Os dados cobrem 10 tópicos de Python: listas, dicionários, funções lambda, exceções, OOP, decorators, generators, arquivos, pathlib e regex. Cada par segue o formato:

```json
{
  "prompt": "Como usar list comprehension em Python?",
  "response": "List comprehension é uma forma concisa de criar listas..."
}
```

### Passo 2 — Quantização com bitsandbytes

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
```

O tipo `nf4` (NormalFloat 4-bit) foi projetado especificamente para pesos de redes neurais, que seguem uma distribuição normal. Ele distribui os 16 valores possíveis de forma que o erro de arredondamento seja mínimo.

### Passo 3 — Configuração do LoRA

```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

O rank `r=64` define o tamanho das matrizes de decomposição. Quanto maior, mais capacidade de aprendizado — mas também mais memória. O `alpha=16` controla o quanto os novos pesos influenciam nas saídas do modelo.

### Passo 4 — Treinamento com SFTTrainer

```python
training_args = SFTConfig(
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    num_train_epochs=3,
    learning_rate=2e-4,
    ...
)
```

O `paged_adamw_32bit` é uma versão do AdamW que move os estados do otimizador para a RAM quando a GPU está sobrecarregada, evitando erros de memória. O scheduler `cosine` faz a taxa de aprendizado diminuir suavemente ao longo do treino, e o `warmup_ratio=0.03` garante que nos primeiros 3% das iterações a taxa suba gradualmente antes de cair.

---

## Hiperparâmetros

| Parâmetro | Valor |
|---|---|
| Modelo | TinyLlama 1.1B |
| Quantização | 4-bit NF4 |
| Compute dtype | float16 |
| LoRA r | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Otimizador | paged_adamw_32bit |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Épocas | 3 |
| Learning rate | 2e-4 |

---

## Resultado

Treinado no Google Colab com GPU T4:

| Métrica | Valor |
|---|---|
| Loss final | 1.169 |
| Acurácia de tokens | 71.77% |
| Tempo total | ~52 segundos |

---

## Referências

- Dettmers et al. (2023) — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Documentação PEFT](https://huggingface.co/docs/peft)
- [Documentação TRL](https://huggingface.co/docs/trl/sft_trainer)
- [TinyLlama no Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)