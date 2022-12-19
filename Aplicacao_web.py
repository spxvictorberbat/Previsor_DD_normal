# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 19:31:52 2022

@author: victor.berbat
"""

import pandas as pd
import numpy as np
import datetime as dt
import streamlit as st

st.write(""" 
         ** Modelo previso de drawdown ** """)
         
        
st.sidebar.header("Escolha seus parâmetros")

sharpe = 0.5
vol = 0.1
anos = 10

Sharpe = float(st.sidebar.text_input("Sharpe", sharpe))
Vol = float(st.sidebar.text_input("Volatilidade", vol))
n = int(st.sidebar.text_input("Anos", anos))

def modelo_normal(sharpe, vol, n, num_simu = 100000):
    drawdowns = np.array([])
    meses = 12*n
    retorno_mensal = ((1 + (sharpe*vol))**(1/12)) - 1
    vol_mensal = vol/np.sqrt(12)
    for i in range(num_simu):
        retornos_simulados = np.random.normal(loc = retorno_mensal, scale = vol_mensal, size = meses)
        retornos_simulados_acumulados = 1*(1 + retornos_simulados).cumprod()
        picos_retornos_simulados_acumulados = np.maximum.accumulate(retornos_simulados_acumulados)
        drawdown_maximo = np.max((picos_retornos_simulados_acumulados - retornos_simulados_acumulados)/picos_retornos_simulados_acumulados)
        drawdowns = np.append(drawdowns, drawdown_maximo)
    
    count, bins_count = np.histogram(drawdowns, bins=150) 
    pdf = count/sum(count) 
    cdf = np.cumsum(pdf)
    mult_sigmas = [0.5, 1, 1.5, 2, 3, 4]
    list_bins_count = list(bins_count)
    
    coluna_x_sigma = [str(i) + 'σ' for i in mult_sigmas]
    Dicionario_probs = {}
    for count, x in enumerate(mult_sigmas):
        try: 
            prob = (1 - cdf[list_bins_count.index(list(filter(lambda i: i > x*vol, list_bins_count))[0])])
        except:
            prob = 0
        Dicionario_probs[coluna_x_sigma[count]] = prob
    
    df_probs = pd.DataFrame(Dicionario_probs.items(), columns = ['Níveis de σ', 'Probabilidades'])
    Dicionario_parametros = {'Sharpe': sharpe, 'Volatilidade': vol, 'Anos': n}
    df_parametros = pd.DataFrame(Dicionario_parametros.items(), columns = ['Parâmetros', 'Valores'])
    
    
    return df_probs

df = modelo_normal(Sharpe, Vol, n)
st.write(df)