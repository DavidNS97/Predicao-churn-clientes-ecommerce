
# ------------------------------------------------------------
# Leia o README.md para explicação do projeto
# ------------------------------------------------------------


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from feature_engine import discretisation,encoding
from sklearn import pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
import numpy as np


# Configurações de visualização para análise exploratória
pd.set_option("display.max_rows", False)
pd.set_option("display.max_columns", False)

# -----------------------------
# Carregamento dos dados
# -----------------------------
df = pd.read_excel("dataset/E Commerce Dataset.xlsx", sheet_name="E Comm")

# -----------------------------
# Visualizando dados e tipos das colunas
# -----------------------------

df.head()
df.dtypes

# -----------------------------
# Ajuste de tipos de dados
# -----------------------------
df['Reclamação'] = df['Reclamação'].astype(bool)
df['Nível da Cidade'] = df['Nível da Cidade'].astype('category')
# %%


# %%
# ============================================================
# 4. Análise de Valores Ausentes
# ============================================================
# As variáveis apresentam baixo percentual de missing values
# (máx. ~5,5%), concentrados em colunas numéricas.
# O volume é considerado seguro para tratamento posterior
# sem impacto relevante na amostra.

missing_df = pd.DataFrame({
    'qtd_vazios': df.isnull().sum(),
    '%vazios': (df.isnull().mean() * 100).round(2)
}).sort_values('%vazios', ascending=False)


# ============================================================
# 5. Feature Engineering
# ============================================================
# Criação de variáveis derivadas para capturar novas features
# Pequeno offset é utilizado para evitar divisão por zero.

df['pedidos_por_ano_rel'] = (
    df['Quantidade de Pedidos'] / (df['Tempo de Relacionamento'] + 0.1)
)

df['rf_score'] = (
    df['Quantidade de Pedidos'] / (df['Dias Desde Último Pedido'] + 0.1)
)

df['intensidade_uso'] = (
    df['Horas no App'] / (df['Quantidade de Pedidos'] + 0.1)
)

df['insatisfacao_recente'] = (
    df['Reclamação'] * (6 - df['Pontuação de Satisfação'])
)

df['distancia_por_pedido'] = (
    df['Armazém até a Casa'] / (df['Quantidade de Pedidos'] + 0.1)
)

df['dispositivos_por_pedido'] = (
    df['Número de Dispositivos Registrados'] / (df['Quantidade de Pedidos'] + 0.1)
)


# ============================================================
# 6. Separação Out-of-Time (OOT)
# ============================================================
# Como o dataset não possui data, utilizamos 'Tempo de Relacionamento'
# como proxy temporal. Clientes mais recentes simulam dados futuros,
# permitindo avaliar acurácia do modelo fora do período de treino e teste.

limite_oot = df['Tempo de Relacionamento'].quantile(0.25)

df_oot = df[df['Tempo de Relacionamento'] <= limite_oot].copy()
df_train = df[df['Tempo de Relacionamento'] > limite_oot].copy()

y_oot = df_oot['Churn']


# ============================================================
# 7. Definição de Features e Target
# ============================================================

target = 'Churn'
features = df_train.columns.drop(target)

X = df_train[features]
y = df_train[target]


# ============================================================
# 8. Split Treino / Teste
# ============================================================
# Divisão com estratificação para manter a taxa de churn
# equivalente entre treino e teste.

from sklearn import model_selection 
X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                                                    X, y,
                                                                    random_state=42, 
                                                                    test_size=0.2, 
                                                                    stratify=y  
                                                                )
# ============================================================
# 9. Verificação de Balanceamento
# ============================================================

print("Taxa de churn geral:", y.mean())
print("Taxa de churn treino:", y_train.mean())
print("Taxa de churn teste:", y_test.mean())

# %%
# ============================================================
# 10. Estatísticas Descritivas por Classe — Variáveis Numéricas
# ============================================================

# Seleciona apenas variáveis numéricas
numericas = X_train.select_dtypes(include=['int64', 'float64']).columns

# DataFrame de análise
df_analise_num = X_train[numericas].copy()
df_analise_num[target] = y_train

# Média e mediana por classe de churn
summario_num = (
    df_analise_num
    .groupby(target)
    .agg(['mean', 'median'])
    .T
)

# Razão relativa entre não churn e churn
summario_num['diff_rel'] = summario_num[0] / summario_num[1]

# Ordena pelas mais discriminativas
summario_num = summario_num.sort_values('diff_rel', ascending=False)

summario_num
# ============================================================
# 11. Matriz de Correlação — Variáveis Numéricas
# ============================================================

corr_matrix = df_analise_num[numericas].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    annot=False,        # evita poluição visual
    linewidths=0.5
)
plt.title("Matriz de Correlação das Variáveis Numéricas", fontsize=14)
plt.show()
# %%

# ============================================================
# 12. Estatísticas Descritivas — Variáveis Categóricas
# ============================================================

# Seleciona variáveis categóricas e booleanas
categoricas = X_train.select_dtypes(
    include=['object', 'category', 'bool']
).columns

df_analise_cat = X_train[categoricas].copy()
df_analise_cat[target] = y_train

# Análise por variável categórica
for col in categoricas:
    
    resumo_cat = (
        pd.crosstab(
            df_analise_cat[col],
            df_analise_cat[target],
            normalize='index'
        ) * 100
    )
    
    resumo_cat = resumo_cat.rename(
        columns={0: '%Nao_Churn', 1: '%Churn'}
    )
    
    # Poder discriminativo relativo
    resumo_cat['diff_rel'] = resumo_cat['%Churn'] / resumo_cat['%Nao_Churn']
    
    resumo_cat = resumo_cat.sort_values('diff_rel', ascending=False)
    
    print(f"\nVariável: {col}")
    print(resumo_cat.round(2))

# %%
# ============================================================
# 13. Remoção de variáveis que não agregam pro modelo
# ============================================================

# ID do cliente é apenas identificador e não agrega poder preditivo
X_train = X_train.drop(columns=["ID do Cliente"])
X_test  = X_test.drop(columns=["ID do Cliente"])
df_oot  = df_oot.drop(columns=["ID do Cliente"])

# ============================================================
# 14. Imputação de valores ausentes 
# ============================================================


# Identifica colunas com missing (apenas colunas numericas possui missing)
numericas = X_train.select_dtypes(include=['int64', 'float64']).columns

colunas_numericas_com_missing = (
    X_train[numericas]
    .columns[X_train[numericas].isna().sum() > 0]
    .tolist()
)

# Imputação pela mediana (boa performance para lidar com outliers)
imputer = SimpleImputer(strategy='median')

# Ajuste apenas no treino (evita vazamento de informação)
X_train[colunas_numericas_com_missing] = imputer.fit_transform(
    X_train[colunas_numericas_com_missing]
)

# Aplicação nos demais conjuntos
X_test[colunas_numericas_com_missing] = imputer.transform(
    X_test[colunas_numericas_com_missing]
)
df_oot[colunas_numericas_com_missing] = imputer.transform(
    df_oot[colunas_numericas_com_missing]
)

# ============================================================
# 15. Dummies Encoding das variáveis categóricas
# ============================================================

X_train = pd.get_dummies(X_train, columns=categoricas, drop_first=False)
X_test  = pd.get_dummies(X_test,  columns=categoricas, drop_first=False)
df_oot  = pd.get_dummies(df_oot,  columns=categoricas, drop_first=False)

# Alinha colunas entre treino, teste e OOT
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
df_oot = df_oot.reindex(columns=X_train.columns, fill_value=0)

# ============================================================
# 16. Inclusão da variável alvo no conjunto OOT
# ============================================================

df_oot['Churn'] = y_oot

# ============================================================
# 17. Seleção de features com Árvore de Decisão
# ============================================================

# Modelo de arvore de decisão para avaliar importância das variáveis
arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

# ============================================================
# 18. Importância das features
# ============================================================

features_importances = (
    pd.Series(arvore.feature_importances_, index=X_train.columns)
      .sort_values(ascending=False)
      .reset_index()
)

features_importances.columns = ['feature', 'importance']
features_importances['importance_acumulada'] = features_importances['importance'].cumsum()

features_importances

# Seleciona features responsáveis por 95% da importância total
best_features = features_importances[
    features_importances['importance_acumulada'] < 0.96
]['feature'].tolist()

best_features


# %%
# ============================================================
# ETAPA 19 — PRÉ-PROCESSAMENTO PARA RANDOM FOREST
# ============================================================
# Discretização supervisionada + OneHot
# ------------------------------------------------------------
# Aqui tratamos variáveis numéricas relevantes usando árvore de decisão
# para discretização, reduzindo sensibilidade a outliers e capturando
# padrões não lineares.
# ------------------------------------------------------------

# ------------------------------------------------------------
# Seleção das variáveis numéricas mais relevantes
# ------------------------------------------------------------

best_features_numericas = [
    col for col in best_features if col in numericas
]

best_features_numericas
# %%


# ------------------------------------------------------------
# Discretização supervisionada
# ------------------------------------------------------------

tree_discretization = discretisation.DecisionTreeDiscretiser(
                        variables=best_features_numericas,
                        regression=False,
                        bin_output='bin_number',
                        cv=3
)
# %%


# ------------------------------------------------------------
# One-Hot Encoding dos bins
# ------------------------------------------------------------

onehot = encoding.OneHotEncoder(
            variables=best_features_numericas,
            ignore_format=True
)
# %%

# ------------------------------------------------------------
# Definição do modelo Random Forest
# ------------------------------------------------------------

model = ensemble.RandomForestClassifier(
            random_state=42,
            min_samples_leaf=20,
            n_jobs=2,
            n_estimators=500
)

params = {
    "min_samples_leaf": [15, 20, 25, 30, 50],
    "n_estimators": [100, 200, 500, 1000],
    "criterion": ['gini', 'entropy', 'log_loss']
}

grid = model_selection.GridSearchCV(
            model,
            params,
            cv=3,
            scoring='roc_auc',
            verbose=4
)
# %%


# ------------------------------------------------------------
# Pipeline do modelo
# ------------------------------------------------------------

model_pipeline = pipeline.Pipeline(
    steps=[
        ("Discretizar", tree_discretization),
        ("onehot", onehot),
        ("Grid", grid)
    ]
)

# ------------------------------------------------------------
#Treinamento do Pipeline com Random Forest  + (Discretização + OneHot + GridSearchCV)
# ------------------------------------------------------------

model_pipeline.fit(X_train[best_features], y_train)

# ------------------------------------------------------------
# Visualizar os melhores hiperparâmetros
# ------------------------------------------------------------

print(grid.best_params_)

# %%
# ============================================================
# ETAPA 20 — MODELO: REGRESSÃO LOGÍSTICA
# ============================================================
# Modelo linear para classificação binária, usado como baseline
# e comparação com modelos não lineares.


# ------------------------------------------------------------
# Definição do modelo regressão logistica
# ------------------------------------------------------------

log_model = linear_model.LogisticRegression(
    solver='liblinear',
    max_iter=500,
    random_state=42
)


# ------------------------------------------------------------
# Grid de hiperparâmetros
# ------------------------------------------------------------

log_params = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10, 100]
}

log_grid = model_selection.GridSearchCV(
    estimator=log_model,
    param_grid=log_params,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=4
)
# %%


# ------------------------------------------------------------
# Pipeline completo
# ------------------------------------------------------------

log_pipeline = pipeline.Pipeline(
    steps=[
        ("Discretizar", tree_discretization),
        ("OneHot", onehot),
        ("Grid", log_grid)
    ]
)

log_pipeline.fit(X_train[best_features], y_train)


# ------------------------------------------------------------
# Melhores hiperparâmetros
# ------------------------------------------------------------

print("Melhores parâmetros da regressão logística:")
print(log_grid.best_params_)


# %%
# ============================================================
# ETAPA 21 — AVALIAÇÃO DOS MODELOS
# ============================================================

from sklearn import metrics

# ------------------------------------------------------------
# Avaliação — Random Forest (Pipeline principal)
# ------------------------------------------------------------

# Treino
y_train_predict = model_pipeline.predict(X_train[best_features])
y_train_proba   = model_pipeline.predict_proba(X_train[best_features])[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)

print("Acurácia treino:", acc_train)
print("AUC treino:", auc_train)

# %%
# Teste
y_test_predict = model_pipeline.predict(X_test[best_features])
y_test_proba   = model_pipeline.predict_proba(X_test[best_features])[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
roc_test = metrics.roc_curve(y_test, y_test_proba)

print("Acurácia teste:", acc_test)
print("AUC teste:", auc_test)

# %%
# OOT (base futura)
y_oot_predict = model_pipeline.predict(df_oot[best_features])
y_oot_proba   = model_pipeline.predict_proba(df_oot[best_features])[:,1]

acc_oot = metrics.accuracy_score(df_oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(df_oot[target], y_oot_proba)
roc_oot = metrics.roc_curve(df_oot[target], y_oot_proba)

print("Acurácia OOT:", acc_oot)
print("AUC OOT:", auc_oot)

# %%
# ------------------------------------------------------------
# Avaliação — Regressão Logística
# ------------------------------------------------------------

# Treino
y_train_predict = log_pipeline.predict(X_train[best_features])
y_train_proba   = log_pipeline.predict_proba(X_train[best_features])[:,1]

acc_train = metrics.accuracy_score(y_train, y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)
roc_train = metrics.roc_curve(y_train, y_train_proba)

# Teste
y_test_predict = log_pipeline.predict(X_test[best_features])
y_test_proba   = log_pipeline.predict_proba(X_test[best_features])[:,1]

acc_test = metrics.accuracy_score(y_test, y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
roc_test = metrics.roc_curve(y_test, y_test_proba)

# OOT
y_oot_predict = log_pipeline.predict(df_oot[best_features])
y_oot_proba   = log_pipeline.predict_proba(df_oot[best_features])[:,1]

acc_oot = metrics.accuracy_score(df_oot[target], y_oot_predict)
auc_oot = metrics.roc_auc_score(df_oot[target], y_oot_proba)
roc_oot = metrics.roc_curve(df_oot[target], y_oot_proba)

print("Acurácia treino:", acc_train)
print("AUC treino:", auc_train)
print("Acurácia teste:", acc_test)
print("AUC teste:", auc_test)
print("Acurácia OOT:", acc_oot)
print("AUC OOT:", auc_oot)

# %%
# ------------------------------------------------------------
# Importância das features — Regressão Logística
# ------------------------------------------------------------

# Melhor modelo encontrado no grid
best_log_model = log_pipeline.named_steps["Grid"].best_estimator_
# %%
# Coeficientes do modelo
coef = best_log_model.coef_[0]

# Gera a  features final (bins + dummies) exatamente como o modelo enxerga
X_transformado = log_pipeline[:-1].transform(X_train[best_features])

# Nomes das features finais
feature_names = X_transformado.columns

# Checagem de segurança
assert len(feature_names) == len(coef), "Mismatch entre features e coeficientes"

# Organização das importâncias
importancia_features = pd.DataFrame({
    "feature": feature_names,
    "coeficiente": coef,
    "importancia_absoluta": np.abs(coef)
}).sort_values("importancia_absoluta", ascending=False)

importancia_features

# %%
# ============================================================
# ETAPA 22 — Plotando curva ROC
# ============================================================

plt.plot(roc_train[0], roc_train[1]) 
plt.plot(roc_test[0], roc_test[1]) 
plt.plot(roc_oot[0], roc_oot[1]) 
plt.plot([0,1] , [0,1], "--", color ='black')
plt.title("Curva ROC")
plt.ylabel("Sensibilidade")
plt.xlabel("1- Especificidade")
plt.legend([ 
            f"Treino = {100*auc_train:.2f}",
            f"Teste = {100*auc_test:.2f}",
            f"Out of Time = {100*auc_oot:.2f}",
            ])
plt.grid(True)
plt.show()
# %%
# ============================================================
# ETAPA 23 — Salvando modelo e features serializados
# ============================================================

#

model_df = pd.Series( {
    "model": log_pipeline,
    "features": best_features,

})
model_df.to_pickle("modelchurn.pkl")

