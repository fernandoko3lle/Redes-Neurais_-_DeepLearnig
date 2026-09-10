# Notas — Atividade 1 (Data)

Arquivo de consulta pessoal, **fora do site** (não faz parte de `docs/`, não é uma entrega).
Resume o que foi feito nesta sessão e o que ainda falta antes de submeter.

## O que foi feito

### Configuração do template (para a edição 2026.2)
- `docs/index.md`: edição marcada `2026.2`; e-mail e GitHub preenchidos na identificação.
- `mkdocs.yml`: `site_url`, `repo_url`, `repo_name` e o link do Colab apontando para
  `fernandoko3lle/Redes-Neurais_-_DeepLearnig`.
- Nome completo preenchido: **Fernando Ganzer Koelle** (`docs/index.md` e `site_author` em
  `mkdocs.yml`).

### Exercise 1 — Nuvens de pontos 2D
Script: `docs/exercises/data/code/exercise1_point_clouds.py` (reescrito do zero).
Roda sozinho, sem dependências externas. Gera Figuras 1–3 e todos os números do enunciado.

Números obtidos (seed 42):

| Item | Valor |
|---|---|
| Taxa de mistura `s=0.5` | 0,25% |
| Taxa de mistura `s=1.0` | 7,25% |
| Taxa de mistura `s=2.0` | 19,25% |
| Taxa de mistura `s=4.0` | 48,25% |
| Menor `r_ij` em `s=1.0` | 1,326 (par Classe 0–1) |
| Menor `r_ij` projetado em `s=2.0` | 0,663 |

### Exercise 2 — Não-linearidade em 5D
Script: `docs/exercises/data/code/exercise2_nonlinearity.py` (novo).
Roda sozinho. Gera Figuras 4–5.

Números obtidos (seed 42):

| Item | Dataset I | Dataset II |
|---|---|---|
| Distância entre centros (5D) | 3,228 | 0,267 |
| Variância explicada PC1+PC2 | 65,97% | 43,16% |

**Decisão que precisa de confirmação:** o enunciado dá o raio das cascas concêntricas como
`ρ ~ N(2.0, 0.4)` (e `N(5.0, 0.4)`). Isso pode significar desvio-padrão **ou** variância —
assumi **desvio-padrão** e documentei no código (`RADIUS_STD = 0.4`). Se o enunciado/monitor
quiser variância, troque para `RADIUS_STD = np.sqrt(0.4)` e rode de novo.

### Exercise 3 — Spaceship Titanic
Script: `docs/exercises/data/code/exercise3_preprocessing.py` (novo). **Executado** com o
`train.csv` real (você baixou e eu movi de `./data/train.csv` para o caminho esperado,
`docs/exercises/data/code/data/train.csv`, já coberto pelo `.gitignore`).

Números obtidos:

| Item | Valor |
|---|---|
| Proporção da classe positiva (`Transported`) | 50,36% |
| Split treino/teste | 6.954 / 1.739 (80/20 estratificado) |
| Média / mediana `FoodCourt` (treino, bruto) | 458,08 / 0,0 |
| Shape final (treino / teste) | (6954, 17) / (1739, 17) |
| NaN remanescente | 0 em ambos |
| Min/max após escalonamento (treino / teste) | [-1,0000, 1,0000] / [-1,0000, 1,1383] |

O teste passar levemente de 1,0 é esperado: o `MinMaxScaler` foi ajustado só no treino (sem
vazamento), e algum valor de teste após o `log1p` é maior que o máximo visto no treino.

### Relatório (`docs/exercises/data/index.md`)
Reescrito com os três exercícios completos (Abordagem/Código/Figuras/Análise). Duas mudanças
importantes em relação à primeira versão:

1. A tabela **Results summary** foi **substituída** pela versão real da edição 2026.2
   (conferida na página oficial via HTML bruto, não só resumo) — a que já estava no template
   era de outra edição e pedia itens diferentes (raio médio de casca interna/externa, que não
   existe mais na tabela atual). Todas as 13 linhas estão preenchidas agora.
2. A seção **Análise** de cada exercício foi reorganizada no formato pergunta → resposta: cada
   pergunta do enunciado aparece como citação (`>`) seguida da resposta, na mesma ordem do
   enunciado, para facilitar a conferência.
3. A Figura 1 ganhou o esboço de fronteiras pedido no item C do Exercise 1 (linhas tracejadas
   entre classes vizinhas, calculadas como bissetriz perpendicular entre os centros).

`ai_use` no front matter já está preenchido descrevendo o uso de IA nesta sessão — releia e
ajuste se quiser detalhar mais.

## Pendências antes de entregar

- [ ] **Reler a Análise dos três exercícios** — foi redigida pela IA a partir dos números; a
      disciplina cobra prova oral, então reescreva com suas palavras o que não estiver
      confortável para defender
- [ ] Confirmar a interpretação de `ρ ~ N(2.0, 0.4)` (desvio-padrão vs. variância) com o
      enunciado/monitoria — só afeta o Exercise 2
- [ ] Commit + push (o prazo é o timestamp do último commit que toca a pasta do exercício)

## Comandos úteis

```shell
# ambiente já criado em ./env
./env/Scripts/python.exe -m mkdocs serve -o          # preview local
./env/Scripts/python.exe -m mkdocs build --strict    # validação antes de commitar

./env/Scripts/python.exe docs/exercises/data/code/exercise1_point_clouds.py
./env/Scripts/python.exe docs/exercises/data/code/exercise2_nonlinearity.py
./env/Scripts/python.exe docs/exercises/data/code/exercise3_preprocessing.py  # precisa do train.csv
```
