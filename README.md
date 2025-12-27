# Predição de Churn em E-commerce com Machine Learning
<p align="left">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/STATUS-EM%20ANDAMENTO-orange" alt="Status: Em Andamento"/>
</p>

## 1. Introdução

Este projeto apresenta uma solução completa de **predição de churn de clientes em e-commerce**, utilizando técnicas de **Machine Learning** aplicadas a dados comportamentais e transacionais.

**Churn** trata-se de cancelamento ou abandono do cliente, ou seja, quando um cliente deixa de comprar ou se relacionar com a empresa. Antecipar esse comportamento permite agir de forma proativa para aumentar retenção e reduzir perdas de receita.

O objetivo da solução é estimar a **probabilidade de churn de cada cliente** e para além da análise, disponibilizar essa informação de forma prática, permitindo a execução de ações de retenção com base em diferentes níveis de risco.

Além da modelagem preditiva, todo o pipeline foi desenvolvido seguindo **boas práticas de ciência de dados**, incluindo análise exploratória, engenharia de features, validação out-of-time, seleção de variáveis, comparação de modelos e disponibilização dos resultados em uma aplicação interativa.

👉 **Acesse o aplicativo interativo:**  
https://app-predicao-churn-ecommerce.streamlit.app/ (clique com o botão direito → Abrir em nova guia)

ℹ️ *Observação: na primeira execução o aplicativo pode levar alguns segundos para carregar, pois o ambiente é inicializado sob demanda pelo Streamlit.*

## 2. Dicionário de Dados

| Coluna                                   | Tipo       | Descrição |
|------------------------------------------|------------|-----------|
| ID do Cliente                            | Numérica   | ID único do cliente |
| Churn                                    | Binária    | Indicador de churn (saída do cliente) |
| Tempo de Relacionamento                  | Numérica   | Tempo de relacionamento do cliente com a empresa (anos) |
| Dispositivo de Login Preferido           | Categórica | Dispositivo de login preferido do cliente |
| Nível da Cidade                          | Categórica | Nível da cidade (1 = grandes capitais; 2 = cidades médias; 3 = cidades pequenas) |
| Armazém até a Casa                       | Numérica   | Distância entre o armazém e a casa do cliente (km) |
| Método de Pagamento Preferido            | Categórica | Método de pagamento preferido do cliente |
| Gênero                                   | Categórica | Gênero do cliente |
| Horas no App                             | Numérica   | Número de horas gastas no aplicativo ou site |
| Número de Dispositivos Registrados       | Numérica   | Total de dispositivos registrados para o cliente |
| Categoria de Pedido Preferida            | Categórica | Categoria de pedido preferida do cliente no último mês |
| Pontuação de Satisfação                  | Numérica   | Pontuação de satisfação do cliente com o serviço |
| Estado Civil                             | Categórica | Estado civil do cliente |
| Número de Endereços                      | Numérica   | Total de endereços adicionados pelo cliente |
| Reclamação                               | Binária    | Se houve alguma reclamação no último mês |
| Aumento do Valor de Pedido vs Ano Anterior | Numérica | Percentual de aumento nos pedidos em relação ao ano anterior |
| Cupons Usados                            | Numérica   | Total de cupons usados no último mês |
| Quantidade de Pedidos                    | Numérica   | Total de pedidos realizados no último mês |
| Dias Desde Último Pedido                 | Numérica   | Dias desde o último pedido do cliente |
| Valor de Cashback                        | Numérica   | Valor médio de cashback no último mês |


## 3. Preparação dos Dados

A etapa de preparação dos dados teve como objetivo garantir a qualidade e a consistência das informações antes da construção do modelo de Machine Learning.

### Carregamento
Os dados foram carregados a partir de um arquivo Excel (`E Commerce Dataset.xlsx`, aba *E Comm*).

## Ajuste de Tipos de Dados

Algumas variáveis vieram com tipos inadequados no dataset original e foram ajustadas para refletir melhor sua natureza:
- **Reclamação** → veio como numérica (0/1), mas foi convertida para **booleano**.  
  *Motivo:* representa apenas presença ou ausência de reclamação, não uma escala numérica.
- **Nível da Cidade** → veio como numérica (1, 2, 3), mas foi convertida para **categórica**.  
  *Motivo:* os valores indicam categorias de cidades (1 = grandes capitais, 2 = cidades médias, 3 = cidades pequenas), não uma ordem contínua.

Esses ajustes garantem que os algoritmos de machine learning interpretem corretamente as variáveis e evitem distorções estatísticas.

### Análise de Valores Ausentes
O objetivo desta etapa foi verificar a qualidade dos dados e identificar variáveis com valores faltantes.  
Essa análise é muito importante, pois:
- **Garante confiabilidade**: valores ausentes podem distorcer estatísticas e comprometer o desempenho dos modelos.  
- **Orienta o tratamento posterior**: ao saber qual percentual de missing em cada variavel, é possível decidir se será feita imputação, exclusão ou outro tipo de ajuste.  
- **Avalia impacto na amostra**: como o percentual máximo de missing foi de ~5,5% e concentrado em variáveis numéricas, consideramos seguro aplicar técnicas de imputação sem prejuízo relevante para o dataset.  

#### Principais variáveis com valores ausentes
| Variável                                   | Qtde Vazios | % Vazios |
|--------------------------------------------|-------------|----------|
| Dias Desde Último Pedido                   | 307         | 5.45%    |
| Aumento do Valor de Pedido vs Ano Anterior | 265         | 4.71%    |
| Tempo de Relacionamento                    | 264         | 4.69%    |
| Quantidade de Pedidos                      | 258         | 4.58%    |
| Cupons Usados                              | 256         | 4.55%    |
| Horas no App                               | 255         | 4.53%    |
| Armazém até a Casa                         | 251         | 4.46%    |



## 4. Feature Engineering

Nesta etapa foram criadas novas variáveis (features) para  capturar padrões adicionais no comportamento dos clientes que não estavam explicitamente representados nas colunas originais.


### Novas variáveis criadas
| Variável                | Fórmula / Origem                                                                 | Objetivo |
|--------------------------|----------------------------------------------------------------------------------|----------|
| **pedidos_por_ano_rel** | `Quantidade de Pedidos / (Tempo de Relacionamento + 0.1)`                        | Frequência real de consumo considerando o tempo de relacionamento |
| **rf_score**            | `Quantidade de Pedidos / (Dias Desde Último Pedido + 0.1)`                       | Recência x Frequência: clientes que compram muito e recentemente têm score maior |
| **intensidade_uso**     | `Horas no App / (Quantidade de Pedidos + 0.1)`                                   | Engajamento de compra no app (diferenciar quem compra de quem só navega) |
| **insatisfacao_recente**| `Reclamação * (6 - Pontuação de Satisfação)`                                     | Combina reclamação recente com baixa satisfação percebida |
| **distancia_por_pedido**| `Armazém até a Casa / (Quantidade de Pedidos + 0.1)`                             | Avaliar impacto da distância logística por pedido |
| **dispositivos_por_pedido** | `Número de Dispositivos Registrados / (Quantidade de Pedidos + 0.1)`         | Relacionar dispositivos vinculados ao volume de pedidos |

Uma soma (+0.1) foi utilizado nas divisões para evitar erros de divisão por zero.


### Impacto no modelo
Antes da criação dessas variáveis derivadas, o modelo apresentava **acurácia abaixo de 50%**.  
Após a inclusão das novas features, houve um ganho significativo de desempenho, mostrando que o *feature engineering* foi decisivo para melhorar a capacidade preditiva.

## 5. Estratégia de Validação

Para garantir que o modelo fosse avaliado de forma robusta, adotamos uma estratégia de validação em múltiplos níveis:

### 5.1 Separação Out-of-Time (OOT)
- Objetivo: avaliar se o modelo mantém desempenho em cenários fora do período de treino e teste, ou seja, num cenario real, ao receber dados novos o modelo será testado se é capaz de lidar com dados diferentes do passado.
- Como o dataset não possui uma coluna de data, utilizamos **Tempo de Relacionamento** como proxy temporal.  
- Clientes mais recentes (quartil inferior de tempo de relacionamento) foram separados como conjunto **OOT**, simulando clientes novos.  

### 5.2 Definição de Features e Target
- **Target:** `Churn` (indicador de saída do cliente).  
- **Features:** todas as demais variáveis inclusive as criadas na feature engineering.  

### 5.3 Split Treino / Teste
- O conjunto de treino foi dividido da base que sobrou em **treino (80%)** e **teste (20%)**.  
- Utilizamos **estratificação** para manter a taxa de churn equivalente entre os conjuntos.  
- Isso garante que a proporção de clientes churn vs não churn seja preservada.

### 5.4 Verificação de Balanceamento
Após realizar o split entre treino e teste, verificamos se a taxa de churn permaneceu equivalente nos diferentes conjuntos.  
Isso é importante porque:

- **Evita viés**: se o treino tivesse muito mais casos de churn que o teste (ou vice‑versa), o modelo poderia aprender padrões artificiais.  
- **Valida a estratificação**: confirma que a divisão preservou a distribuição da variável alvo assegurando que o modelo seja avaliado em condições próximas às reais.

Resultados:
- Taxa de churn geral: ~5,79%  
- Taxa de churn treino: ~5,81%  
- Taxa de churn teste: ~5,74%
  
As taxas são praticamente iguais, mostrando que o split foi bem sucedido e que o modelo será treinado e avaliado em bases comparáveis.


#### Fluxograma separação dos dados



## 6. Análise Exploratória dos Dados (EDA)

Nesta etapa foram realizadas análises estatísticas e visuais para compreender melhor o comportamento das variáveis e sua relação com o churn.

### 6.1 Estatísticas Descritivas — Variáveis Numéricas
- Média e mediana calculadas por classe (`Churn` vs `Não Churn`).
- Criada a métrica **diff_rel** para identificar variáveis mais discriminativas.

O grafico de barras abaixo mostra as principais variáveis numéricas com suas média por classe de churn e a razão relativa (`diff_rel`).
Valores maiores que 1 indicam que a variável tende a ser maior em clientes **não churn**, enquanto valores menores que 1 indicam maior associação com **churn**.




**Insights principais:**
- Clientes churn tendem a ter **menos dias desde o último pedido** e **menor tempo de relacionamento**.
- Variáveis como **insatisfação recente**, **número de endereços** e **rf_score** aparecem mais altas em clientes churn.
- O uso de **cupons** também é relativamente maior em churners.
  
### 6.2 Matriz de Correlação — Variáveis Numéricas
- Objetivo: identificar relações fortes e possíveis redundâncias.

IMAGEM


**Insights principais da correlação:**
- Algumas variáveis apresentam alta correlação entre si (ex.: Quantidade de Pedidos e Pedidos por Ano), indicando redundância.  
- Variáveis como **Insatisfação Recente** e **Dias Desde Último Pedido** têm correlação mais baixa com as demais features, sugerindo maior poder explicativo isolado para churn.  

### 6.3 Estatísticas Descritivas — Variáveis Categóricas
- Para cada variável categórica, calculada a proporção de churn e não churn.
- Criada a métrica **diff_rel** para medir poder discriminativo relativo.

| Variável                        | Categoria              | %Não Churn | %Churn | diff_rel |
|---------------------------------|------------------------|------------|--------|----------|
| Dispositivo de Login Preferido  | Computer               | 93.69      | 6.31   | 0.07     |
| Dispositivo de Login Preferido  | Mobile Phone           | 94.20      | 5.80   | 0.06     |
| Nível da Cidade                 | 3                      | 90.50      | 9.50   | 0.10     |
| Nível da Cidade                 | 2                      | 97.76      | 2.24   | 0.02     |
| Método de Pagamento Preferido   | E-wallet               | 89.01      | 10.99  | 0.12     |
| Método de Pagamento Preferido   | UPI                    | 98.30      | 1.70   | 0.02     |
| Categoria de Pedido Preferida   | Fashion                | 90.91      | 9.09   | 0.10     |
| Categoria de Pedido Preferida   | Grocery                | 95.87      | 4.13   | 0.04     |
| Estado Civil                    | Single                 | 91.12      | 8.88   | 0.10     |
| Estado Civil                    | Married                | 95.64      | 4.36   | 0.05     |
| Reclamação                      | True                   | 87.70      | 12.30  | 0.14     |
| Reclamação                      | False                  | 96.58      | 3.42   | 0.04     |

**Insights principais:**
- **Maior churn** em clientes com **reclamações**, que usam **E-wallet**,  e compram **Fashion/Mobile**.  
- **Menor churn** em clientes que usam **UPI**, compram **Laptop & Accessory** e não registram reclamações.


## 7. Preparação para Modelagem

### 7.1 Remoção de variáveis irrelevantes
- A coluna **ID do Cliente** foi removida dos conjuntos de treino, teste e OOT por se tratar de  apenas de um identificador único, sem poder preditivo, e poderia induzir o modelo a padrões artificiais.

### 7.2 Imputação de valores ausentes
-Após identificar as variáveis com valores faltantes na etapa de "Preparação dos Dados" chegou a hora de tratar esses valores nulos, pois não são aceitos nos principais algoritmos de ML
- Apenas colunas **numéricas** apresentavam valores faltantes.  
- Foi utilizada a **mediana** para imputação, pois é sólida contra outliers, ou seja, valores extremos não distorcem o preenchimento e preserva a distribuição das variáveis numéricas.
- O calculo foi  feito considerando  **apenas no conjunto de treino** para evitar vazamento de informação ja que o modelo de teste e OOT são dados "novos", e considerar-los poderia enviesar o modelo.
- A transformação foi aplicada posteriormente nos conjuntos de teste e OOT.

### 7.3 Codificação de variáveis categóricas
- As variáveis categóricas foram convertidas em **dummies (one-hot encoding)** para que o modelo consiga interpretar corretamente categorias não numéricas. Esse processo evita que o algoritmo crie relações artificiais entre categorias (como se houvesse uma ordem entre elas) e garante que cada categoria seja representada de forma independente em colunas binárias (0/1). 
- Após a codificação, os conjuntos de treino, teste e OOT foram **reindexados** para garantir que todos possuíssem as mesmas colunas para garantir consistência estrutural e comparabilidade, para que assim o modelo consiga aplicar o mesmo raciocínio em qualquer dado novo.

#### Fluxograma da preparação dos dados

## 8. Seleção das Melhores Features

Após a preparação dos dados, foi necessário selecionar as variáveis mais relevantes para o modelo de churn.  Essa etapa é importante pois é nela que reduz a dimensionalidade dos dados para manter apenas variáveis que realmente contribuem para a previsão.
Embora a **Análise Exploratória (EDA)** seja útil para identificar correlações e padrões, ela não captura totalmente a **importância preditiva** das variáveis.  

Por isso, gosto de  utilizar  **árvore de decisão** para calcular a importância das features.  
Esse método tem como vantagens:
- Considera interações não lineares entre variáveis.  
- Avalia o impacto direto de cada feature na redução da impureza dos nós da árvore.  
- É mais completo que apenas observar correlações, já que algumas variáveis podem ser pouco correlacionadas isoladamente, mas altamente relevantes em conjunto.  

### Processo
1. Calculamos a importância de cada variável com base no modelo de árvore nos dados de **TREINO**.  
2. Ordenamos as features por importância.  
3. Criamos uma coluna de **importância acumulada**.  
4. Selecionamos as variáveis responsáveis por **95% da importância total**.  

### Resultado
As variáveis selecionadas (best_features) representam o subconjunto mais relevante para explicar o churn, .


## 9 Random Forest

O algoritmo **Random Forest** foi testado pela sua  capacidade de lidar com variáveis numéricas e categóricas, além de  combinar resultados de múltiplas árvores de decisão para produzir previsões mais estáveis e generalizáveis
Para garantir o melhor desempenho, aplicamos uma busca sistemática de parâmetros (GridSearchCV).

### O que foi feito
- Definimos um conjunto de **parâmetros candidatos** (como número de árvores, tamanho mínimo das folhas e critério de divisão).
- O **GridSearchCV** testou todas as combinações possíveis desses parâmetros.
- Cada combinação foi avaliada com **validação cruzada (cv=3)** usando a métrica **ROC AUC**.
- O processo escolheu automaticamente a configuração que apresentou o melhor resultado.

### Parâmetros testados
Foram avaliadas diferentes combinações de parâmetros para encontrar o melhor equilíbrio entre **complexidade do modelo** e **capacidade de generalização**:
- `min_samples_leaf`: [15, 20, 25, 30, 50]     - Relacionado ao número mínimo de amostras necessárias para que um nó folha seja criado
- `n_estimators`: [100, 200, 500, 1000]        - Relacionado ao número de modelos individuais (árvores de decisão) que serão construídos
- `criterion`: ['gini', 'entropy', 'log_loss'] - Define  a função de avaliação usada para escolher os melhores splits nas árvores.

### Resultado
O modelo final foi treinado com os **melhores parâmetros encontrados**, garantindo maior capacidade preditiva e menor risco de overfitting.

### Pipeline
Foi criado um **pipeline** integrando o modelo Random Forest com o GridSearchCV, garantindo organização e reprodutibilidade do processo de treinamento.

### Treinamento
O pipeline foi aplicado nos dados de treino, utilizando as **best_features** e os **parâmetros otimizados** pelo GridSearchCV.

## 10 Regressão Logística

A **Regressão Logística** foi utilizada como modelo baseline por sua simplicidade e interpretabilidade ( esse modelo em especifico, pude estudar muito no periodo da faculdade)
Assim como na Random Forest, aplicamos a  otimização de parâmetros e para melhorar a representação das variáveis contínuas, aplicamos etapas de discretização, algo que não é necessário na arvore de decisao.

### O que foi feito

- **Discretização supervisionada:** as variáveis numéricas contínuas foram transformadas em bins (intervalos) usando árvores de decisão.  
  Essa técnica divide os valores em faixas que ajudam o modelo a aprender melhor, permitindo capturar padrões não lineares, reduzir o impacto de outliers e facilitar a interpretação.  
  - Sem a discretização, o modelo perdeu aproximadamente **22% de poder discriminativo (AUC)** e **15% de acurácia no OOT**.
- **One-Hot Encoding:** os bins resultantes foram convertidos em variáveis categóricas binárias.  
  Essa etapa é importante porque evita que os intervalos sejam interpretados como valores numéricos contínuos, garantindo que cada faixa seja tratada como uma categoria independente.  
- O GridSearchCV  testou todas as combinações possíveis dos  parâmetros selecionados 
- Cada combinação foi avaliada com **validação cruzada (cv=3)** usando **ROC AUC**, e selecionamos automaticamente a configuração com melhor desempenho.

### Parâmetros testados
- `penalty`: ["l1", "l2"] - tipo de regularização aplicada.  
- `C`: [0.01, 0.1, 1, 10, 100] - controla a força da regularização (valores menores = regularização mais forte).  

### Pipeline
Foi criado um **pipeline completo**, integrando:
1. Discretização supervisionada  
2. One-Hot Encoding  
3. GridSearchCV com regressão logística  

### Treinamento
O pipeline foi aplicado nos dados de treino, utilizando as **best_features** e os **parâmetros otimizados** pelo GridSearchCV.

## Avaliação dos Modelos
- Acurácia
- ROC AUC
- Curva ROC
- Avaliação em treino, teste e OOT

## Serialização do Modelo
- Salvamento do modelo
- Salvamento das features

## Aplicação Prática (Streamlit)
(Como o modelo é utilizado na prática)

## Tecnologias Utilizadas






