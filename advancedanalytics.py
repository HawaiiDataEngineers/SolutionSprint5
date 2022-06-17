from google.colab import auth
auth.authenticate_user()
print('Authenticated')

#Instalando pacotes e implementando bibliotecas

import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as mp
import numpy as np
import statistics as st

dfAnalise = pd.io.gbq.read_gbq('''
  SELECT  * FROM `fiap-challenge-5.resultado_final.ds_covid_pr` as covid
  INNER JOIN `fiap-challenge-5.resultado_final.ds_sales_iowa_srw` as sales
  ON sales.date = covid.data_notificacao 
''', project_id='fiap-challenge-5', dialect='standard')

dfAnalise =  dfAnalise[['date', 'ocupacao_CovidUti', 'item_description','bottles_sold','ocupacao_CovidCli']]

#Aplicando filtros no dataset para o produto Old Overholt Rye
dfAnalise = dfAnalise.loc[dfAnalise.item_description == 'Old Overholt Rye']

#Informações do dataset
dfAnalise.info()

dfAnalise.corr()

#Cáculo das medidas de tendêncial central 
print("****************************************************************************************************************************")
print("Cáculo das medidas de tendência central do dataset Covid no Paraná e Empresa Liquor Sales na cidade de IOWA no Ano de 2020 *")
print("****************************************************************************************************************************")
print("")
#Cáculo da média leitos UTI e clínicos ocupados por COVID
print("Média leitos UTI ocupados por COVID:",st.mean(dfAnalise.ocupacao_CovidUti))
print("Média leitos Clínicos ocupados por COVID:",st.mean(dfAnalise.ocupacao_CovidCli))

#Cáculo da média de garrafas Old Overholt Rye	vendidas pela Empresa Liquor Sales
print("Média de garrafas Old Overholt Rye vendidas:",st.mean(dfAnalise.bottles_sold))
print("")

#Cáculo da mediana de leitos UTI e clínicos ocupados por COVID
print("Mediana dos leitos UTI ocupados por COVID:",st.median(dfAnalise.ocupacao_CovidUti))
print("Mediana dos leitos Clínicos ocupados por COVID:",st.median(dfAnalise.ocupacao_CovidCli))

#Cáculo da mediana de garrafas Old Overholt Rye	vendidas
print("Mediana de garrafas Old Overholt Rye vendidas:",st.median(dfAnalise.bottles_sold))
print("")

#Cáculo da moda de leitos UTI e clínicos ocupados por COVID
print("Moda dos leitos UTI ocupados por COVID:",st.mode(dfAnalise.ocupacao_CovidUti))
print("Moda dos leitos Clínicos ocupados por COVID:",st.mode(dfAnalise.ocupacao_CovidCli))

#Cáculo da moda de garrafas Old Overholt Rye vendidas
print("Moda de garrafas Old Overholt Rye vendidas:",st.mode(dfAnalise.bottles_sold))
print("")

#Cálculo dos quartis de leitos UTI e clínicos ocupados por COVID
print("1º quartil dos leitos UTI ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidUti, 0.25))
print("2º quartil dos leitos UTI ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidUti, 0.50))
print("3º quartil dos leitos UTI ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidUti, 0.75))
print("")
print("1º quartil dos leitos Clínicos ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidCli, 0.25))
print("2º quartil dos leitos Clínicos ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidCli, 0.50))
print("3º quartil dos leitos Clínicos ocupados por COVID:",np.quantile(dfAnalise.ocupacao_CovidCli, 0.75))
print("")

#Cálculo dos quartis de garrafas Old Overholt Rye vendidas
print("1º quartil de garrafas Old Overholt Rye vendidas:",np.quantile(dfAnalise.bottles_sold, 0.25))
print("2º quartil de garrafas Old Overholt Rye vendidas:",np.quantile(dfAnalise.bottles_sold, 0.50))
print("3º quartil de garrafas Old Overholt Rye vendidas:",np.quantile(dfAnalise.bottles_sold, 0.75))
print("")

#Cáculo das medidas de dispersão de leitos UTI e clínicos ocupados por COVID
print("******************************************************************************************************************")
print("Cáculo das medidas de dispersão do dataset Covid no Paraná e Empresa Liquor Sales na cidade de IOWA no Ano de 2020")
print("******************************************************************************************************************")
print("")
#varianca populacional de leitos UTI e clínicos ocupados por COVID
print("Variância populacional dos leitos UTI ocupados por COVID:",st.pvariance(dfAnalise.ocupacao_CovidUti))
print("Variância populacional dos leitos Clínicos ocupados por COVID:",st.pvariance(dfAnalise.ocupacao_CovidCli))

#varianca populacional de garrafas Old Overholt Rye vendidas
print("Variância populacional de garrafas Old Overholt Rye vendidas:",st.pvariance(dfAnalise.bottles_sold))
print("")

#desvio padrao de leitos UTI e clínicos ocupados por COVID
print("Desvio padrao dos leitos UTI ocupados por COVID:",st.pstdev(dfAnalise.ocupacao_CovidUti))
print("Desvio padrao dos leitos Clínicos ocupados por COVID:",st.pstdev(dfAnalise.ocupacao_CovidCli))

#desvio de garrafas Old Overholt Rye vendidas
print("Desvio padrao de garrafas Old Overholt Rye vendidas:",st.pstdev(dfAnalise.bottles_sold))
print("")

#desvio padrao amostral de leitos UTI e clínicos ocupados por COVID
print("Desvio padrao amostral dos leitos UTI ocupados por COVID:",st.stdev(dfAnalise.ocupacao_CovidUti))
print("Desvio padrao amostral dos leitos Clínicos ocupados por COVID:",st.stdev(dfAnalise.ocupacao_CovidCli))

#desvio padrao amostral de garrafas Old Overholt Rye vendidas
print("Desvio padrao amostral de garrafas Old Overholt Rye vendidas:",st.stdev(dfAnalise.bottles_sold))
print("")

#amplitude(a) de leitos UTI e clínicos ocupados por COVID
print("Amplitude(a) dos leitos UTI ocupados por COVID:",max(dfAnalise.ocupacao_CovidUti)- min(dfAnalise.ocupacao_CovidUti))
print("Amplitude(a) dos leitos Clínicos ocupados por COVID:",max(dfAnalise.ocupacao_CovidCli)- min(dfAnalise.ocupacao_CovidCli))

#amplitude(a) de garrafas Old Overholt Rye vendidas
print("Amplitude(a) de mortes sofridas pelo time blue:",max(dfAnalise.bottles_sold)- min(dfAnalise.bottles_sold))
print("")

#Intervalo interquartil de leitos UTI e clínicos ocupados por COVID
print("Intervalo interquartil dos leitos UTI ocupados por COVID:",np.percentile(dfAnalise.ocupacao_CovidUti, 75)-np.percentile(dfAnalise.ocupacao_CovidUti, 25))
print("Intervalo interquartil dos leitos Clínicos ocupados por COVID:",np.percentile(dfAnalise.ocupacao_CovidCli, 75)-np.percentile(dfAnalise.ocupacao_CovidCli, 25))

#Intervalo interquartil de garrafas Old Overholt Rye vendidas
print("Intervalo interquartil de garrafas Old Overholt Rye vendidas:",np.percentile(dfAnalise.bottles_sold, 75)-np.percentile(dfAnalise.bottles_sold, 25))
print("")


#Gráfico de frequência de ocupação dos leitos UTI ocupados por COVID
print("*************************************************************")
print("Gráfico de frequência de ocupação dos leitos de UTI por COVID")
print("*************************************************************")
print("")
mp.hist(dfAnalise.ocupacao_CovidUti)


#Gráfico de frequência de ocupação dos leitos Clínicos ocupados por COVID
print("************************************************************************")
print("Gráfico de frequência de ocupação dos leitos Clínicos ocupados por COVID")
print("************************************************************************")
print("")
mp.hist(dfAnalise.ocupacao_CovidCli)


#Gráfico de frequência de vendas de garrafas Old Overholt Rye
print("************************************************************")
print("Gráfico de frequência de vendas de garrafas Old Overholt Rye")
print("************************************************************")
print("")
mp.hist(dfAnalise.bottles_sold)

#Teste de normalidade
# Distribuicao de frequencia
#1. Calcular os quartis
print("**************************************************************")
print("Teste de normalidade para ocupação dos leitos de UTI por COVID")
print("**************************************************************")
print("")

quantis = []
for i in range (1,dfAnalise.ocupacao_CovidUti.size+1):
  quantis.append(i/dfAnalise.ocupacao_CovidUti.size)
np.quantile(dfAnalise.ocupacao_CovidUti, quantis)
mp.plot(quantis, dfAnalise.ocupacao_CovidUti.sort_values())


#Teste de normalidade
# Distribuicao de frequencia
#1. Calcular os quartis
print("*************************************************************************")
print("Teste de normalidade para ocupação dos leitos Clínicos ocupados por COVID")
print("*************************************************************************")
print("")

quantis = []
for i in range (1,dfAnalise.ocupacao_CovidCli.size+1):
  quantis.append(i/dfAnalise.ocupacao_CovidCli.size)
np.quantile(dfAnalise.ocupacao_CovidCli, quantis)
mp.plot(quantis, dfAnalise.ocupacao_CovidCli.sort_values())


#Teste de normalidade
# Distribuicao de frequencia
#1. Calcular os quartis
print("***********************************************************")
print("Teste de normalidade de vendas de garrafas Old Overholt Rye")
print("***********************************************************")
print("")

quantis = []
for i in range (1,dfAnalise.bottles_sold.size+1):
  quantis.append(i/dfAnalise.bottles_sold.size)
np.quantile(dfAnalise.bottles_sold, quantis)
mp.plot(quantis, dfAnalise.bottles_sold.sort_values())


#Regressão Linear
model = sm.OLS(dfAnalise.ocupacao_CovidUti , dfAnalise.bottles_sold).fit()
model.summary()

import statistics
statistics.variance(dfAnalise.ocupacao_CovidUti)

statistics.variance(dfAnalise.ocupacao_CovidCli)

statistics.variance(dfAnalise.bottles_sold)
