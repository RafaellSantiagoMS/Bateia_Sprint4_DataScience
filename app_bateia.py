"""
Bateia - Sprint 4 de Data Science
Grupo: Enzo Augusto (RM562249), Gustavo Neres (RM561785),
       Rafaell Santiago (RM563486), Sebastian Iriarte (RM563619)

Executar com: streamlit run app_bateia.py
"""
import re
import string
import unicodedata
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Bateia - análise de reuniões",
    layout="wide",
)


@st.cache_resource
def carregar_arquivos():
    modelo = joblib.load("modelo_oportunidade.joblib")
    tfidf = joblib.load("tfidf_termos.joblib")
    referencias = joblib.load("referencias.joblib")
    return modelo, tfidf, referencias


modelo, tfidf_termos, ref = carregar_arquivos()
vocabulario = tfidf_termos.get_feature_names_out()
paridade = pd.read_csv("paridade.csv")


# =========================================================
# MESMAS FUNÇÕES DO NOTEBOOK
# =========================================================

def limpar_transcricao(texto):
    t = re.sub(r'\[LOCUTOR\s*\d+\]:?', ' ', texto)
    t = re.sub(r'\[(PESSOA|EMPRESA|LOCAL)\]', ' ', t)
    t = t.lower()
    t = re.sub(f'[{re.escape(string.punctuation)}]', ' ', t)
    t = re.sub(r'\b\d+\b', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def pipeline_limpeza(texto):
    palavras = [p for p in limpar_transcricao(texto).split()
                if p not in ref['STOPWORDS_PT'] and len(p) >= 3]
    return ' '.join(palavras)


def remove_acentos(s):
    return ''.join(ch for ch in unicodedata.normalize('NFD', s)
                   if unicodedata.category(ch) != 'Mn')


def texto_para_busca(texto):
    return remove_acentos(limpar_transcricao(texto))


def contar_termos(texto_busca, termos):
    encontrados = {}
    for termo in termos:
        n = len(re.findall(r'\b' + termo + r'\b', texto_busca))
        if n > 0:
            encontrados[termo] = n
    return encontrados


def identificar(texto_busca, dicionario):
    return [nome for nome, variacoes in dicionario.items() if contar_termos(texto_busca, variacoes)]


def trecho(texto_busca, termo):
    # o termo com até 15 palavras antes e depois, como evidência do sinal
    achados = re.findall(r'(?:\S+ ){0,15}\b' + termo + r'\b(?: \S+){0,15}', texto_busca)
    if not achados:
        return ''
    return '...' + achados[0] + '...'


def valores_citados(transcricao):
    return sorted(set(re.findall(r'R\$ ?\d+(?:[.,]\d+)*(?: mil| milhões)?', transcricao)))


def medir_conversa(transcricao):
    texto = re.sub(r'\[(PESSOA|EMPRESA|LOCAL)\]', 'x', transcricao)
    falas = re.findall(r'\[LOCUTOR (\d+)\]:?([^\[]*)', texto)
    palavras_por_locutor = Counter()
    for locutor, fala in falas:
        palavras_por_locutor[locutor] += len(fala.split())

    palavras = texto_para_busca(transcricao).split()
    n_palavras = max(len(palavras), 1)
    perguntas = transcricao.count('?') / n_palavras * 1000

    maiores = palavras_por_locutor.most_common(2)
    total = sum(palavras_por_locutor.values())
    if len(maiores) == 2:
        balanco = 1 - abs(maiores[0][1] / (maiores[0][1] + maiores[1][1]) - 0.5) * 2
    else:
        balanco = 0
    engajamento = 1 - maiores[0][1] / total if total > 0 else 0

    final = ' '.join(palavras[int(len(palavras) * 0.8):])
    proximos = sum(contar_termos(final, ref['PROXIMOS_PASSOS']).values())

    return {'perguntas': perguntas, 'balanco': balanco, 'engajamento': engajamento,
            'proximos_passos': proximos, 'n_locutores': len(palavras_por_locutor)}


def principais_termos(texto_limpo, n=8):
    # mesmo cálculo de pesos da Sprint 2, aplicado a uma reunião só
    pesos = np.asarray(tfidf_termos.transform([texto_limpo]).sum(axis=0)).flatten()
    termos = pd.Series(pesos, index=vocabulario).sort_values(ascending=False).head(n)
    return list(termos[termos > 0].index)


def pontuar_sentimento(texto_busca):
    positivas = sum(contar_termos(texto_busca, ref['PALAVRAS_POSITIVAS']).values())
    negativas = sum(contar_termos(texto_busca, ref['PALAVRAS_NEGATIVAS']).values())
    if positivas + negativas == 0:
        return 0
    return (positivas - negativas) / (positivas + negativas)


def classe_sentimento(pontuacao):
    if pontuacao < ref['SENT_Q1']:
        return 'Negativo'
    if pontuacao > ref['SENT_Q3']:
        return 'Positivo'
    return 'Neutro'


def nota_risco(n_sinais):
    return round(100 * min(n_sinais / ref['LIMITE_CHURN'], 1))


def nivel_risco(nota):
    if nota == 100:
        return 'Alto'
    if nota > 0:
        return 'Atenção'
    return 'Sem sinais'


def notas_ita(medidas_conversa):
    return {c: round(100 * min(medidas_conversa[c] / ref['REFERENCIA_ITA'][c], 1))
            for c in ref['CRITERIOS_ITA']}


def selo_confianca(n_locutores):
    if n_locutores == 0 or n_locutores > ref['LIMITE_OUTLIER']:
        return 'Baixa'
    if n_locutores > ref['Q3_LOCUTORES']:
        return 'Média'
    return 'Alta'


def faixa_oportunidade(nota):
    if nota >= 60:
        return 'Alta'
    if nota >= 40:
        return 'Média'
    return 'Baixa'


def faixa_prioridade(prioridade):
    if prioridade >= 75:
        return 'Crítica'
    if prioridade >= 55:
        return 'Alta'
    if prioridade >= 35:
        return 'Média'
    return 'Baixa'


def gerar_recomendacao(r):
    produtos = ', '.join(r['produtos']) if r['produtos'] else 'a solução discutida'
    if r['nivel_risco'] == 'Alto':
        acao = 'Retenção: acionar o gestor da conta e agendar contato com o cliente em até 48 horas.'
    elif r['faixa_oportunidade'] == 'Alta':
        acao = 'Avançar a venda: enviar proposta ou agendar demonstração de ' + produtos + '.'
    elif r['nivel_risco'] == 'Atenção' and r['sentimento'] == 'Negativo':
        acao = 'Cuidar do relacionamento: retomar os pontos negativos antes de oferecer algo novo.'
    elif r['sinais_expansao'] >= 2:
        acao = 'Oferecer expansão: apresentar módulos ou licenças adicionais ao cliente.'
    elif r['faixa_oportunidade'] == 'Média':
        acao = 'Qualificar: confirmar necessidade, prazo e orçamento na próxima conversa.'
    else:
        acao = 'Acompanhar: manter o relacionamento no ciclo normal da conta.'

    avisos = []
    if r['nivel_risco'] == 'Alto' and r['faixa_oportunidade'] == 'Alta':
        avisos.append('A reunião também parece prospecção: confirme se a insatisfação é com a TOTVS ou com o fornecedor atual do cliente.')
    if r['concorrentes']:
        avisos.append('Concorrente citado (' + ', '.join(r['concorrentes']) + '): leve um comparativo para a próxima conversa.')
    if r['selo'] == 'Baixa':
        avisos.append('Selo de confiança baixo: confira a transcrição antes de agir.')
    return acao, avisos


def resumir(r):
    partes = []
    if r['produtos']:
        partes.append('falou sobre ' + ', '.join(r['produtos'][:3]))
    if r['concorrentes']:
        partes.append('citou ' + ' e '.join(r['concorrentes']))
    if r['nivel_risco'] == 'Alto':
        partes.append('deu sinais de que pode cancelar')
    if r['faixa_oportunidade'] == 'Alta':
        partes.append('tem perfil de oportunidade de venda')
    if not partes:
        return 'Nesta conversa não apareceram sinais fortes de risco nem de oportunidade.'
    return 'Nesta conversa, o cliente ' + '; '.join(partes) + '.'


def analisar_reuniao(transcricao):
    texto_limpo = pipeline_limpeza(transcricao)
    texto_busca = texto_para_busca(transcricao)
    conversa = medir_conversa(transcricao)

    indice = list(modelo.classes_).index('Oportunidade')
    oportunidade = round(100 * modelo.predict_proba([texto_limpo])[0][indice])

    sinais = contar_termos(texto_busca, ref['SINAIS_CHURN'])
    risco = nota_risco(sum(sinais.values()))
    criterios = notas_ita(conversa)
    concorrentes = identificar(texto_busca, ref['CONCORRENTES'])

    r = {
        'produtos': identificar(texto_busca, ref['PRODUTOS_TOTVS']),
        'concorrentes': concorrentes,
        'trechos_concorrentes': [trecho(texto_busca, ref['CONCORRENTES'][c][0]) for c in concorrentes],
        'valores': valores_citados(transcricao),
        'sentimento': classe_sentimento(pontuar_sentimento(texto_busca)),
        'risco': risco,
        'nivel_risco': nivel_risco(risco),
        'sinais_churn': sinais,
        'trechos_churn': [trecho(texto_busca, s) for s in list(sinais)[:3]],
        'oportunidade': oportunidade,
        'faixa_oportunidade': faixa_oportunidade(oportunidade),
        'sinais_expansao': sum(contar_termos(texto_busca, ref['SINAIS_EXPANSAO']).values()),
        'prioridade': round(0.62 * risco + 0.38 * oportunidade),
        'ita': round(np.mean(list(criterios.values()))),
        'criterios_ita': criterios,
        'selo': selo_confianca(conversa['n_locutores']),
        'n_locutores': conversa['n_locutores'],
        'principais_termos': principais_termos(texto_limpo),
    }

    pior = min(criterios, key=criterios.get)
    if criterios[pior] < 50:
        r['conducao'] = ref['DICAS_CONDUCAO'][pior]
    else:
        r['conducao'] = 'Boa condução: descoberta, equilíbrio de fala e encaminhamento presentes.'
    r['resumo'] = resumir(r)
    r['recomendacao'], r['avisos'] = gerar_recomendacao(r)
    return r


# =========================================================
# TRANSCRIÇÃO
# =========================================================

st.title("Bateia - o que esta reunião diz para o time comercial?")
st.write(
    "Cole a transcrição de uma reunião (com as marcações [LOCUTOR n]) ou escolha um exemplo "
    "na barra lateral. A análise usa o mesmo modelo e as mesmas regras do notebook da Sprint 4."
)

st.sidebar.header("Exemplos")
exemplo = st.sidebar.selectbox("Carregar reunião de exemplo",
                               ["Nenhum", "exemplo_oportunidade.txt", "exemplo_risco.txt"])
texto_inicial = ""
if exemplo != "Nenhum":
    with open(exemplo, encoding="utf-8") as f:
        texto_inicial = f.read()

with st.form("formulario"):
    transcricao = st.text_area("Transcrição da reunião", value=texto_inicial, height=220)
    enviar = st.form_submit_button("Analisar reunião")


# =========================================================
# RESULTADO
# =========================================================

if enviar and len(transcricao.strip()) < 50:
    st.info("Cole uma transcrição com pelo menos algumas falas.")
elif enviar:
    r = analisar_reuniao(transcricao)

    st.subheader("Recomendação de ação")
    st.success(r["recomendacao"])
    for aviso in r["avisos"]:
        st.warning(aviso)
    st.write(r["resumo"])

    st.subheader("Indicadores (0 a 100)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Oportunidade comercial", str(r["oportunidade"]) + " - " + r["faixa_oportunidade"])
    col2.metric("Risco de churn", str(r["risco"]) + " - " + r["nivel_risco"])
    col3.metric("Prioridade", str(r["prioridade"]) + " - " + faixa_prioridade(r["prioridade"]))
    col4.metric("ITA", r["ita"])

    col1, col2 = st.columns(2)
    col1.markdown("**Produto identificado:** " + (", ".join(r["produtos"]) or "nenhum"))
    col1.markdown("**Sentimento:** " + r["sentimento"])
    col1.markdown("**Valores citados:** " + (", ".join(r["valores"]) or "nenhum"))
    col1.markdown("**Principais termos:** " + ", ".join(r["principais_termos"]))
    col1.markdown("**Selo de confiança:** " + r["selo"] + " (" + str(r["n_locutores"]) + " locutores)")
    col2.markdown("**Condução da reunião (ITA)**")
    col2.dataframe(pd.DataFrame({"Critério": [ref["CRITERIOS_ITA"][c] for c in r["criterios_ita"]],
                                 "Nota": list(r["criterios_ita"].values())}), hide_index=True)
    col2.write(r["conducao"])

    if r["concorrentes"] or r["sinais_churn"]:
        with st.expander("Trechos da conversa"):
            for nome, t in zip(r["concorrentes"], r["trechos_concorrentes"]):
                st.write("**Concorrente " + nome + ":** " + t)
            for sinal, t in zip(list(r["sinais_churn"]), r["trechos_churn"]):
                st.write("**Sinal de churn (" + sinal + "):** " + t)


# =========================================================
# TESTE DE PARIDADE
# =========================================================

st.subheader("Teste de paridade")
st.write(
    "Reunião de exemplo analisada no notebook. O app aplica as mesmas funções e o mesmo modelo "
    "salvo; os números precisam ser iguais."
)
with open(paridade["arquivo"][0], encoding="utf-8") as f:
    r_app = analisar_reuniao(f.read())

comparacao = pd.DataFrame({
    "Notebook": [paridade["oportunidade"][0], paridade["risco"][0], paridade["ita"][0], paridade["prioridade"][0]],
    "App": [r_app["oportunidade"], r_app["risco"], r_app["ita"], r_app["prioridade"]],
}, index=["Oportunidade", "Risco", "ITA", "Prioridade"])
st.dataframe(comparacao)
