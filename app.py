import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

#streamlit run app.py

# Importanto modelo de regressão logistica
model_df = pd.read_pickle("modelchurn.pkl")
model = model_df['model']
features = model_df['features']


##Leitura dataset de clientes
df_lista_clientes = pd.read_excel("dataset/E Commerce Dataset.xlsx", sheet_name="E Comm")
#selecionando apenas clientes ativos
df_lista_clientes = df_lista_clientes[df_lista_clientes['Churn'] == 0]

##Criando colunas igual no modelo 
pedido_preferido_opp= ['Laptop & Accessory','Mobile','Mobile Phone','Others','Fashion','Grocery']
metodo_pagamento_opp = ['Debit Card','UPI','CC','Cash on Delivery','E wallet','COD','Credit Card']
disp_login_opp= ['Computer','Phone','Mobile Phone']
Nível_da_Cidade_opp= ['1','2','3']


#Variaveis dummy do modelo
dummy_vars=['Dispositivo de Login Preferido', 'Nível da Cidade',
        'Método de Pagamento Preferido',
        'Categoria de Pedido Preferida',]

#features do modelo
df_template =pd.DataFrame(columns=[
        'Armazém até a Casa',
        'Número de Endereços',
        'Tempo de Relacionamento',
        'Aumento do Valor de Pedido vs Ano Anterior',
        'distancia_por_pedido',
        'dispositivos_por_pedido',
        'pedidos_por_ano_rel',
        'Pontuação de Satisfação',
        'insatisfacao_recente',
        'Valor de Cashback',
        'Categoria de Pedido Preferida_Laptop & Accessory',
        'Dias Desde Último Pedido',
        'rf_score',
        'Categoria de Pedido Preferida_Mobile',
        'Método de Pagamento Preferido_Cash on Delivery',
        'Quantidade de Pedidos',
        'Dispositivo de Login Preferido_Mobile Phone',
        'intensidade_uso',
        'Nível da Cidade_3',
        'Horas no App',
        'Dispositivo de Login Preferido_Phone'
    ])
##

#Definindo titulo e icone da pagina
st.set_page_config(page_title='Predição Churn', page_icon='🔍')

#Mensagem home

st.markdown("""
# 👋 Boas-vindas!

## Preditor de Churn para E-commerce
Este aplicativo utiliza **Machine Learning** para estimar a probabilidade de churn de cada cliente.  
Com base no nivel de risco, aplicamos **regras de negócio** para sugerir ações práticas de retenção.

🔎 **Saiba mais sobre o projeto:**  
[Explicação completa do código](https://github.com/DavidNS97/Predicao-churn-clientes-ecommerce)  

💼 **Conecte-se comigo:**  
[LinkedIn](https://www.linkedin.com/in/davidnunes9/)

""")


#Exibição simulador de churn 
exp2 = st.expander("Clientes e Probabilidade de Churn")
with exp2:
    # --- COPIAR AS TRANSFORMAÇÕES EXATAS DO TREINO ---
    # Alteração de tipo de variável — 
    df_lista_clientes['Reclamação'] = df_lista_clientes['Reclamação'].astype(bool)
    df_lista_clientes['Nível da Cidade'] = df_lista_clientes['Nível da Cidade'].astype('category')
    numericas = df_lista_clientes.select_dtypes(include=['int64', 'float64']).columns
    categoricas = df_lista_clientes.select_dtypes(include=['object','category','bool']).columns

    # Imputação de valores ausentes (missing) em colunas numéricas

    colunas_numericas_com_missing = df_lista_clientes[numericas].columns[df_lista_clientes[numericas].isna().sum() > 0].tolist()
    #Substituindo por média

    imputer = SimpleImputer(strategy='median')

    # Substituição dos valores missing 
    df_lista_clientes[colunas_numericas_com_missing] = imputer.fit_transform(df_lista_clientes[colunas_numericas_com_missing])

    #  Criar features novas igual no treino
    df_lista_clientes["distancia_por_pedido"] = df_lista_clientes["Armazém até a Casa"] / (df_lista_clientes["Quantidade de Pedidos"] + 0.1)
    df_lista_clientes["dispositivos_por_pedido"] = df_lista_clientes["Número de Dispositivos Registrados"] / (df_lista_clientes["Quantidade de Pedidos"] + 0.1)
    df_lista_clientes["pedidos_por_ano_rel"] = df_lista_clientes["Quantidade de Pedidos"] / (df_lista_clientes["Tempo de Relacionamento"] + 0.1)
    df_lista_clientes["insatisfacao_recente"] = df_lista_clientes['Reclamação'] * (6 - df_lista_clientes['Pontuação de Satisfação'])
    df_lista_clientes["intensidade_uso"] = df_lista_clientes["Horas no App"] / (df_lista_clientes["Quantidade de Pedidos"] + 0.1)
    df_lista_clientes['rf_score'] = df_lista_clientes['Quantidade de Pedidos'] / (df_lista_clientes['Dias Desde Último Pedido'] + 0.1)
    #Dummy variaveis categoricas
    df_lista_clientes_numericos = df_lista_clientes.drop(columns=dummy_vars)
    df_dummy_lista_cliente =pd.get_dummies(df_lista_clientes[dummy_vars], drop_first=False).astype(int)
    df_final_clientes = pd.concat([df_lista_clientes_numericos, df_dummy_lista_cliente], axis=1)
    # garante que todas as colunas do template existam
    df_final_clientes = df_final_clientes.reindex(columns=df_template.columns, fill_value=0)

    # Fazer predição de probabilidade para cada cliente

    predicao = model.predict_proba(df_final_clientes)[:, 1]
    #Novas colunas
    df_final_clientes["Probabilidade Churn (%)"] = (predicao * 100).round(0).astype(int)
    df_final_clientes["ID do Cliente"] = df_lista_clientes["ID do Cliente"]
    proba_churn= df_final_clientes["Probabilidade Churn (%)"]
    conditions = [
        proba_churn >= 85,                      # muito alto
        (proba_churn >= 70) & (proba_churn < 85),     # alto
        (proba_churn >= 30) & (proba_churn < 70),     # moderado
        proba_churn < 30                        # baixo
    ]
    #Definição das ações pra cada % de probabiliadade de churn ( regra de negocio)
    actions = [
        "Enviar cupom agressivo",
        "Oferecer cashback+",
        "Sugerir produtos relacionados",
        "Monitorar"
    ]
    #Coluna de ação recomendada

    df_final_clientes["Ação Recomendada"] = np.select(conditions, actions, default="Monitorar")


    #exibir tabela:
    colunas_exibir = [
        "ID do Cliente",
        "Probabilidade Churn (%)",
        "Ação Recomendada"
    ]

    st.dataframe(
        df_final_clientes[colunas_exibir],
        use_container_width=True,
        column_config={
            "Probabilidade Churn (%)": st.column_config.ProgressColumn(
                "Probabilidade Churn (%)",
                help="Probabilidade prevista pelo modelo",
                format="%d%%",
                min_value=0,
                max_value=100
            )
        }
    )


#Exibição da lista de clientes no app 
exp1 = st.expander("Simulação individual")
with exp1:
    col1, col2, col3 = st.columns(3)
    #Criação dos inputs de acordo com as features do modelo

    with col1:
        distancia =  st.number_input("Distancia em Km do armazém até a casa:",0,200)
        Número_dispositivo =  st.number_input("Número de Dispositivos Registrados:",1,6)
        Número_endereços =  st.number_input("Número de Endereços Registrados:",1,22)
        Tempo_Relacionamento =  st.number_input("Tempo Relacionamento do Cliente (anos):",0,60)
        horas_app =  st.number_input("Média semanal horas gastas no app:",0,6)
    with col2:
        Qtd_Pedido =  st.number_input("Quantidade de pedidos realizados no último mês:",0,16)
        Aumento_pedido =  st.number_input("% De aumento pedido vs ano anterior:",0,100)
        Satisfação =  st.number_input("Pontuação de Satisfação com serviço:",1,5)
        Cashback =  st.number_input("Valor médio (R$) de cashback no último mês:",0,400)
        dias =  st.number_input("Dias desde o último pedido do cliente:",0,200)
    with col3:
        pedido_preferido =  st.selectbox("Categoria de pedido preferida do cliente no último mês:",pedido_preferido_opp)
        meio_pagamento= st.selectbox("Método de pagamento preferido :",metodo_pagamento_opp)
        login_preferido= st.selectbox("Dispositivo de login preferido:",disp_login_opp)
        nivel_cidade = st.selectbox("Nível da cidade que reside o cliente:",Nível_da_Cidade_opp,help="1 = Grandes capitais / cidades principais\n"
                                                                                        "2 = Cidades médias\n"
                                                                                        "3 = Cidades pequenas ou menores"
                                                                            )
        Reclamação = st.selectbox("Houve reclamação ultimo mês:",['Sim','Não'])
        reclamacao_num = 1 if Reclamação == "Sim" else 0

    #Criando dataset com as features do modelo e a respectiva resposta apontada na simulação
    data = {
        'Armazém até a Casa':distancia,
        'Número de Dispositivos Registrados':Número_dispositivo,
        'Número de Endereços':Número_endereços,
        'Tempo de Relacionamento':Tempo_Relacionamento,
        'Aumento do Valor de Pedido vs Ano Anterior':Aumento_pedido,
        'Pontuação de Satisfação':Satisfação,
        'Valor de Cashback':Cashback,
        'Categoria de Pedido Preferida':pedido_preferido,
        'Dias Desde Último Pedido':dias,
        'Método de Pagamento Preferido':meio_pagamento,
        'Quantidade de Pedidos':Qtd_Pedido,
        'Dispositivo de Login Preferido':login_preferido,
        'Nível da Cidade':nivel_cidade,
        'Horas no App':horas_app,
        'distancia_por_pedido':distancia/(Qtd_Pedido+ 0.1),
        'dispositivos_por_pedido':Número_dispositivo/(Qtd_Pedido+ 0.1),
        'pedidos_por_ano_rel':Qtd_Pedido/(Tempo_Relacionamento+ 0.1),
        'insatisfacao_recente':reclamacao_num * ( 6-Satisfação ),
        'rf_score': Qtd_Pedido/(dias+ 0.1),
        'intensidade_uso':horas_app/(Qtd_Pedido+ 0.1),
    }
    df = pd.DataFrame([data]).replace({"Sim":1, "Não":0})

    #Dummy variaveis categoricas

    df_numericos = df.drop(columns=dummy_vars)

    df_dummy =pd.get_dummies(df[dummy_vars], drop_first=False).astype(int)
    df_final = pd.concat([df_numericos, df_dummy], axis=1)
    # garante que todas as colunas do template existam
    df_final = df_final.reindex(columns=df_template.columns, fill_value=0)

    #probabilidade
    proba = model.predict_proba(df_final[features])[:, 1]


    proba_value = float(proba[0])

    if proba_value < 0.20:
        cor = "#2ecc71"  # verde
        texto = "Cliente com baixo risco de churn"
    elif proba_value < 0.50:
        cor = "#f1c40f"  # amarelo
        texto = "Cliente com risco moderado de churn"
    else:
        cor = "#e74c3c"  # vermelho
        texto = "Cliente com alto risco de churn"

    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:10px;
            background-color:{cor};
            text-align:center;
            color:white;
            font-size:24px;
            font-weight:bold;">
            {proba_value:.1%} — {texto}
        </div>
        """,
        unsafe_allow_html=True
    )