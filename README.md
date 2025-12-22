# Predição de Churn em E-commerce com Machine Learning
<p align="left">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
  <img src="https://img.shields.io/badge/STATUS-EM%20ANDAMENTO-orange" alt="Status: Em Andamento"/>
</p>

## 1. Introdução

Este projeto apresenta uma solução completa de **predição de churn de clientes em e-commerce**, utilizando técnicas de **Machine Learning** aplicadas a dados comportamentais e transacionais.

**Churn** refere-se ao cancelamento ou abandono do cliente, ou seja, quando um cliente deixa de comprar ou se relacionar com a empresa. Antecipar esse comportamento permite agir de forma proativa para aumentar retenção e reduzir perdas de receita.

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

### 6.1 Separação Out-of-Time (OOT)
- Objetivo: avaliar se o modelo mantém desempenho em cenários fora do período de treino e teste, ou seja, num cenario real, ao receber dados novos o modelo será testado se é capaz de lidar com dados diferentes do passado.
- Como o dataset não possui uma coluna de data, utilizamos **Tempo de Relacionamento** como proxy temporal.  
- Clientes mais recentes (quartil inferior de tempo de relacionamento) foram separados como conjunto **OOT**, simulando clientes novos.  

### 6.2 Definição de Features e Target
- **Target:** `Churn` (indicador de saída do cliente).  
- **Features:** todas as demais variáveis inclusive as criadas na feature engineering.  

### 6.3 Split Treino / Teste
- O conjunto de treino foi dividido da base que sobrou em **treino (80%)** e **teste (20%)**.  
- Utilizamos **estratificação** para manter a taxa de churn equivalente entre os conjuntos.  
- Isso garante que a proporção de clientes churn vs não churn seja preservada.

### 6.4 Verificação de Balanceamento
Após realizar o split entre treino e teste, verificamos se a taxa de churn permaneceu equivalente nos diferentes conjuntos.  
Isso é importante porque:

- **Evita viés**: se o treino tivesse muito mais casos de churn que o teste (ou vice‑versa), o modelo poderia aprender padrões artificiais.  
- **Valida a estratificação**: confirma que a divisão preservou a distribuição da variável alvo assegurando que o modelo seja avaliado em condições próximas às reais.

Resultados:
- Taxa de churn geral: ~5,79%  
- Taxa de churn treino: ~5,81%  
- Taxa de churn teste: ~5,74%
  
As taxas são praticamente iguais, mostrando que o split foi bem sucedido e que o modelo será treinado e avaliado em bases comparáveis.


### Esquema Visual
Um diagrama simples ajuda a entender a separação:



## Análise Exploratória dos Dados (EDA)
- Estatísticas descritivas (numéricas e categóricas)
- Matriz de correlação


- 
## Preparação para Modelagem
- Imputação de valores ausentes
- Criação de variáveis dummy
- Padronização do dataset final

## Seleção das Melhores Features
- Árvore de decisão
- Corte em 95% de importância acumulada

## Modelagem
- Random Forest
- Regressão Logística

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

