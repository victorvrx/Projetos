#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[7]:


boston_df


# In[8]:


df=boston_df


# In[9]:


# Configurando o estilo do seaborn
sns.set(style="whitegrid")

# Boxplot para o valor médio das residências ocupadas pelos proprietários (MEDV)
plt.figure(figsize=(8, 6))
sns.boxplot(x='MEDV', data=df, orient='v')
plt.title('Boxplot do Valor Médio das Residências Ocupadas pelos Proprietários (MEDV)')
plt.xlabel('Valor Médio das Residências (MEDV)')
plt.show()


# In[10]:


# Gráfico de barras para a variável Charles River (CHAS)
plt.figure(figsize=(6, 4))
sns.countplot(x='CHAS', data=df)
plt.title('Gráfico de Barras para a Variável Charles River (CHAS)')
plt.xlabel('Margem do Rio Charles (1 se delimita, 0 caso contrário)')
plt.ylabel('Contagem')
plt.show()


# In[11]:


# Boxplot para MEDV versus AGE (discretizada em três grupos)
plt.figure(figsize=(10, 6))
df['AGE_group'] = pd.cut(df['AGE'], bins=[0, 35, 70, max(df['AGE'])], labels=['35 anos ou menos', 'Entre 35 e 70 anos', '70 anos ou mais'])
sns.boxplot(x='AGE_group', y='MEDV', data=df)
plt.title('Boxplot de MEDV versus Idade (Discretizada)')
plt.xlabel('Grupo de Idade')
plt.ylabel('Valor Médio das Residências (MEDV)')
plt.show()


# In[12]:


# Gráfico de dispersão para NOX versus INDUS
plt.figure(figsize=(8, 6))
sns.scatterplot(x='NOX', y='INDUS', data=df)
plt.title('Gráfico de Dispersão para NOX versus INDUS')
plt.xlabel('Concentração de Óxidos Nítricos (NOX)')
plt.ylabel('Proporção de Acres de Negócios Não Varejistas (INDUS)')
plt.show()


# In[13]:


# Histograma para a variável PTRATIO
plt.figure(figsize=(8, 6))
sns.histplot(df['PTRATIO'], bins=20, kde=True)
plt.title('Histograma da Proporção Aluno-Professor (PTRATIO)')
plt.xlabel('Proporção Aluno-Professor (PTRATIO)')
plt.ylabel('Frequência')
plt.show()


# In[14]:


from scipy import stats

river_charles = df[df['CHAS'] == 1]['MEDV']
no_river_charles = df[df['CHAS'] == 0]['MEDV']

t_statistic, p_value = stats.ttest_ind(river_charles, no_river_charles)

alpha = 0.05

if p_value < alpha:
    print(f'Rejeitamos a hipótese nula. Existe uma diferença significativa no valor médio das casas delimitadas pelo rio Charles e aquelas que não são.')
else:
    print('Não há evidências suficientes para rejeitar a hipótese nula. Não há diferença significativa no valor médio das casas.')


# In[15]:


from scipy.stats import f_oneway

age_groups = [df[df['AGE_group'] == group]['MEDV'] for group in df['AGE_group'].unique()]

f_statistic, p_value_anova = f_oneway(*age_groups)

if p_value_anova < alpha:
    print('Rejeitamos a hipótese nula. Existe uma diferença significativa no valor médio das residências para diferentes grupos de idade.')
else:
    print('Não há evidências suficientes para rejeitar a hipótese nula. Não há diferença significativa no valor médio das residências para diferentes grupos de idade.')


# In[16]:


correlation, p_value_corr = stats.pearsonr(df['NOX'], df['INDUS'])

if p_value_corr < alpha:
    print('Rejeitamos a hipótese nula. Existe uma correlação significativa entre as concentrações de óxido nítrico e a proporção de acres de negócios não varejistas.')
else:
    print('Não há evidências suficientes para rejeitar a hipótese nula. Não há correlação significativa entre as concentrações de óxido nítrico e a proporção de acres de negócios não varejistas.')


# In[20]:


import statsmodels.api as sm

# Adicionando uma constante ao conjunto de dados
X = sm.add_constant(df['DIS'])
y = df['MEDV']

# Ajustando o modelo de regressão linear
model = sm.OLS(y, X).fit()

# Obtendo os resultados do modelo
results = model.summary()

# Extraindo o p-valor para a variável DIS
p_value_dis = float(results.tables[1].data[1][3])

if p_value_dis < alpha:
    print('Rejeitamos a hipótese nula. A distância ponderada para os cinco centros de emprego tem um impacto significativo no valor médio das residências.')
else:
    print('Não há evidências suficientes para rejeitar a hipótese nula. A distância ponderada para os cinco centros de emprego não tem impacto si')


# In[ ]:




