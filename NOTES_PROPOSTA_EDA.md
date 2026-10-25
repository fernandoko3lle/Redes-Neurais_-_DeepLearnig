# Proposta de dataset — formulário do EDA

Preencher em: https://forms.gle/zREqauuojy8Phvbb7

Envie os 3, em ordem de preferência. Nenhum é "pré-aprovado" — o professor escolhe um.

## 1ª opção — Stroke Prediction Dataset (classificação)

- **URL:** https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **Tamanho:** 5.110 linhas × 12 colunas
- **Alvo:** `stroke` (binário: teve AVC ou não)
- **Features:** 3 numéricas (`age`, `avg_glucose_level`, `bmi`) + 7 categóricas (`gender`,
  `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`,
  `smoking_status`) — `id` é identificador, descartar.
- **Motivação:** problema real de risco de saúde, mistura genuína de features numéricas e
  categóricas, e o alvo é fortemente desbalanceado (~5% positivo) — obriga a equipe a lidar
  com desbalanceamento em vez de um classificador trivial de maioria.
- **Primeiro risco:** desbalanceamento severo da classe positiva; `bmi` tem valores faltantes.

## 2ª opção — Adult Income / Census Income (classificação)

- **URL:** https://www.kaggle.com/datasets/uciml/adult-census-income
- **Tamanho:** ~48.842 linhas × 15 colunas
- **Alvo:** `income` (binário: `<=50K` / `>50K`)
- **Features:** 6 numéricas (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`,
  `hours-per-week`) + 8 categóricas (`workclass`, `education`, `marital-status`, `occupation`,
  `relationship`, `race`, `sex`, `native-country`).
- **Motivação:** amostra grande, categóricas de alta cardinalidade (`native-country`,
  `occupation`), e a presença de atributos sensíveis (`race`, `sex`) traz uma discussão real
  sobre vazamento/justiça no pré-processamento.
- **Primeiro risco:** valores faltantes codificados como `"?"` em `workclass`, `occupation` e
  `native-country` (não aparecem como `NaN` direto).

## 3ª opção — Telco Customer Churn (classificação)

- **URL:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Tamanho:** 7.043 linhas × 21 colunas
- **Alvo:** `Churn` (binário: `Yes`/`No`)
- **Features:** 3 numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) + 16 categóricas
  (serviços contratados, tipo de contrato, forma de pagamento, etc.).
- **Motivação:** problema de negócio (retenção de clientes), estrutura quase toda categórica
  com muitas colunas binárias/multi-categoria — testa o pipeline de encoding a sério.
- **Primeiro risco:** `TotalCharges` vem como texto (`object`) com valores em branco para
  clientes com `tenure = 0` — precisa de conversão e tratamento antes de virar numérica.
