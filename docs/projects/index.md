# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Se os nomes não estiverem aqui, não há
    como atribuir a nota — e o mesmo vale para o `mkdocs.yml`, cujo `site_author` deve
    listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Fernando Ganzer Koelle | fgkoelle@gmail.com | [@fernandoko3lle](https://github.com/fernandoko3lle) |

Times de 2 a 3 pessoas — por ora só eu, atualize se entrarem colegas. Repita esses nomes no
cabeçalho de cada entrega — quem corrige pode abrir uma página sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Classificação | [Classificação](classification/index.md) |
| 3 | Generativo | [Generativo](generative/index.md) |

Datas e pesos são da sua edição — veja o
[overview](https://insper.github.io/ann-dl/){:target='_blank'}.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Deliverables bem escritos não sustentam uma equipe que não consegue explicar o que
    entregou. Escreva os relatórios de modo que você consiga defendê-los meses depois, e
    confira no overview da sua edição como a prova entra na nota.

## Dataset

O mesmo dataset atravessa as três entregas — escolhê-lo bem no EDA é o que torna as outras
duas viáveis.

!!! success "Aprovado"

    Dataset validado pelo professor a partir da proposta enviada pelo
    [formulário da disciplina](https://forms.gle/zREqauuojy8Phvbb7){:target='_blank'}.

| | |
|---|---|
| **Nome** | Stroke Prediction Dataset |
| **Fonte (URL)** | <https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset> |
| **Licença / termos de uso** | Kaggle — copyright dos autores, uso educacional |
| **Amostras** | 5.110 |
| **Features** | 10 (3 numéricas, 7 categóricas) + `id` (identificador, descartado) |
| **Variável alvo** | `stroke` (binária: teve AVC ou não) |
| **Tarefa escolhida** | Classificação |

Dataset de risco de AVC: mistura genuína de features numéricas (`age`, `avg_glucose_level`,
`bmi`) e categóricas (histórico de hipertensão, doença cardíaca, tabagismo, tipo de trabalho,
etc.), e o alvo é fortemente desbalanceado (só ~5% dos casos são positivos) — o que torna a
tarefa interessante: um classificador "sempre não" já acerta ~95%, então o desafio real é lidar
com esse desbalanceamento, não só treinar um modelo qualquer.

## Status

- [ ] **1. EDA**
- [ ] **2. Classificação**
- [ ] **3. Generativo**

## Registro de decisões

Anote aqui as decisões que atravessam as entregas — troca de dataset, mudança de alvo,
recorte de features — com a data. É o que permite reconstruir o raciocínio na prova de
projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 2026-10-06 | Escolhida tarefa de Classificação; proposto Stroke Prediction como 1ª opção de dataset (Adult Income e Telco Customer Churn como reservas) | Perfil do dataset (mix numérico/categórico, alvo desbalanceado) citado como exemplo pelo enunciado |
| 2026-10-25 | Stroke Prediction aprovado pelo professor | Confirmação via formulário da disciplina |
| | | |
