# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 10:45:01 2025

@author: alexa
"""
import streamlit as st
import numpy as np
from scipy.optimize import differential_evolution

def f(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p):
    f_forte = (beta_forte / eta_forte) * ((t / eta_forte) ** (beta_forte - 1)) * np.exp(-((t / eta_forte) ** beta_forte))
    f_fraco = (beta_fraco / eta_fraco) * ((t / eta_fraco) ** (beta_fraco - 1)) * np.exp(-((t / eta_fraco) ** beta_fraco))
    return p * f_fraco + (1 - p) * f_forte

def F(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p):
    F_forte = 1 - np.exp(-((t / eta_forte) ** beta_forte))
    F_fraco = 1 - np.exp(-((t / eta_fraco) ** beta_fraco))
    return p * F_fraco + (1 - p) * F_forte

def R(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p):
    R_forte = np.exp(-((t / eta_forte) ** beta_forte))
    R_fraco = np.exp(-((t / eta_fraco) ** beta_fraco))
    return p * R_fraco + (1 - p) * R_forte

def V_neg_log(parametros, DCp, DCE, DCD):
    p, beta_fraco, r_eta_fraco, beta_forte, eta_forte = parametros
    eta_fraco = r_eta_fraco * eta_forte
    Vs = 0
    if DCp:
        Vs += sum(np.log10(f(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p)) for t in DCp)
    if DCE:
        Vs += sum(np.log10(F(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p)) for t in DCE)
    if DCD:
        Vs += sum(np.log10(R(t, beta_forte, eta_forte, beta_fraco, eta_fraco, p)) for t in DCD)
    return -Vs

st.title("Distribuição Mista")

p_chute = st.number_input("Estimativa inicial do percentual de itens fracos (%):", min_value=0.0, max_value=100.0, key="p_chute")

# Verifica se há dados antes de converter para float
def processar_entrada(texto):
    return [float(x.strip()) for x in texto.split(',') if x.strip()]

DCp_texto = st.text_area("Dados completos (separados por vírgula)", key="DCp")
DCE_texto = st.text_area("Dados censurados à esquerda (separados por vírgula)", key="DCE")
DCD_texto = st.text_area("Dados censurados à direita (separados por vírgula)", key="DCD")

DCp = processar_entrada(DCp_texto)
DCE = processar_entrada(DCE_texto)
DCD = processar_entrada(DCD_texto)


if st.button("Estimar Parâmetros"):
    vetor_comum = np.array(DCp + DCE + DCD)
    eta_max = 100 * max(vetor_comum) if len(vetor_comum) > 0 else 10000
    p_inicial = p_chute / 100
    bounds = [(max(p_inicial - 0.05, 0), p_inicial + 0.05), (0, 10), (0, 1), (0, 10), (0, eta_max)]
    
    res = differential_evolution(lambda x: V_neg_log(x, DCp, DCE, DCD), bounds)
    par = res.x
    ver = -res.fun
    
    st.write(f"Percentual estimado de itens fracos: {par[0] * 100:.2f}%")
    st.write("### Para os itens fracos:")
    st.write(f"Parâmetro de escala: {par[2] * par[4]:.4f}")
    st.write(f"Parâmetro de forma: {par[1]:.4f}")
    st.write("### Para os itens fortes:")
    st.write(f"Parâmetro de escala: {par[4]:.4f}")
    st.write(f"Parâmetro de forma: {par[3]:.4f}")
