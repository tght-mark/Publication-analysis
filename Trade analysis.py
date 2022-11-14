#!/usr/bin/env python
# coding: utf-8

# In[266]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install statsmodels')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install numpy')
get_ipython().system('pip install linearmodels')



# In[267]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from linearmodels import RandomEffects


# In[268]:


data = pd.read_csv("aggregate_analysis.csv")


# In[269]:


data


# In[270]:


data.plot(x = 'LN_GDPCHN', y = 'LN_Trade', kind = 'scatter', c = 'r')


# In[271]:


data.plot(x = 'LN_GDPPartner', y = 'LN_Trade', kind = 'scatter', c = 'b')


# In[272]:


dep_var1 = data["LN_GDPCHN"]
dep_var2 = data["LN_GDPPartner"]
exp_var1 = data["LN_Trade"]


# In[273]:


exp_var1 = sm.add_constant(exp_var1)
exp_var1


# -TRADE ON CHINA GDP
# 

# In[274]:


model1 = sm.OLS(dep_var1, exp_var1)
result = model1.fit()
print(result.summary())


# Insights
# - No reportable significance,

# ##How other factors of growth affect GDPCHN

# In[275]:


exp_var5 = data[["LN_Distance","LN_PPOP","LN_EXRATE","TFREE_DIFF","COV19D","LNGDPPCD"]]
dep_var6 = data["LN_GDPCHN"]

exp_var5 = sm.add_constant(exp_var5)
exp_var5

model6 = sm.OLS(dep_var6, exp_var5)
result = model6.fit()
print(result.summary()) 


# Insights
# - A 1percent increase in occurence of covid, decreased the GDP growth OF China by -0.62%
# - A 1percent increase in difference in trade freedom between both countries, decreased the GDP growth OF China by -0.53%

# -Trade on EU GDP
# 

# In[276]:


model2 = sm.OLS(dep_var2, exp_var1)
result = model2.fit()
print(result.summary())


# Insights
# - A 1percent increase in trade, increased the GDP OF EU by 0.4percent

# ##Effect of other factors of growth on EU GDP

# In[277]:


exp_var6 = data[["LN_Distance","LN_PPOP","LN_EXRATE","TFREE_DIFF","COV19D","LNGDPPCD"]]
dep_var7 = data["LN_GDPPartner"]

exp_var6 = sm.add_constant(exp_var6)
exp_var6

model7 = sm.OLS(dep_var7, exp_var6)
result = model7.fit()
print(result.summary()) 


# Insights
# - A 1percent increase in occurence of covid, increased the GDP OF EUpartners by 1.5percent
# 

#  #Using fixedeffects

# In[278]:


Country = data["Country"]
Year = pd.Categorical(data["Year"])
data = data.set_index(["Country", "Year"])
data["Year"] = Year

dep_var8 = data["LN_GDPCHN"]
dep_var9 = data["LN_GDPPartner"]
exp_var7 = data["LN_Trade"]

exp_var7 = sm.add_constant(exp_var7)
exp_var7


# - Trade on China GDP
# 

# In[279]:


model8 = PanelOLS(dep_var8, exp_var7, entity_effects=True)
res = model8.fit()
print(res)


# Insights
# - A 1percent increase in TRADE, increased the GDP OF CHINA by -0.6percent

# - TRADE ON EU GDP

# In[280]:


model9 = PanelOLS(dep_var9, exp_var7, entity_effects=True)
res = model9.fit()
print(res)


# Insights
# - A 1percent increase in TRADE, increased the GDP OF EU by 1.5percent

# - Other factors of trade on gdp china
# 

# In[282]:


exp_var10 = data[["LN_Distance","LN_PPOP","LN_EXRATE","TFREE_DIFF","COV19D","LNGDPPCD"]]
dep_var8 = data["LN_GDPCHN"]

exp_var10 = sm.add_constant(exp_var10)
exp_var10

model10 = PanelOLS(dep_var8, exp_var10, entity_effects=True)
res = model10.fit()
print(res)


# Insights
# - A 1percent increase in Difference in TRADE Freedom between both countries, reduced the GDP OF CHINA by -1.2 percent
# - A 1percent increase in Covid occurence between both countries, reduced the GDP OF CHINA by -1.04 percent

# - Other factors of economic growth on gdp EU

# In[290]:


exp_var10 = data[["LN_Distance","LN_PPOP","LN_EXRATE","TFREE_DIFF","COV19D","LNGDPPCD"]]
dep_var9 = data["LN_GDPPartner"]

exp_var10 = sm.add_constant(exp_var10)
exp_var10

model12 = PanelOLS(dep_var9, exp_var10, entity_effects=True)
res = model12.fit()
print(res)


# Insights
# 
# A 1percent increase in exchange rate between both countries, increased the GDP OF EU by 0.2 percent
# A 1percent increase in Covid occurence between both countries, increased the GDP OF EU by 1.6 percent

#  Other factors of economic growth on trade

# In[286]:


exp_var10 = data[["LN_Distance","LN_PPOP","LN_EXRATE","TFREE_DIFF","COV19D","LNGDPPCD"]]
dep_var10 = data["LN_Trade"]

exp_var10 = sm.add_constant(exp_var10)
exp_var10

model12 = PanelOLS(dep_var10, exp_var10, entity_effects=True)
res = model12.fit()
print(res)


# Insights
# 
# A 1percent increase in covid occurence, increased the trade by 0.4 percent
# A 1percent increase in demand differences between both countries, decreased trade by -0.19 percent

# In[ ]:




