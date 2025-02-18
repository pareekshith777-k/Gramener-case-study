#!/usr/bin/env python
# coding: utf-8

# # Lending Club Default Analysis
# 
# The analysis is divided into four main parts:
# 1. Data understanding 
# 2. Data cleaning (cleaning missing values, removing redundant columns etc.)
# 3. Data Analysis 
# 4. Recommendations
# 

# In[185]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[186]:


loan=pd.read_csv("loan.csv",sep=',')
loan.head()


# ### Data understaning

# In[187]:


loan.info()


# In[188]:


loan.columns


# In[189]:


loan.describe()


# Some of the important columns in the dataset are loan_amount, term, interest rate, grade, sub grade, annual income, purpose of the loan etc.
# 
# The **target variable**, which we want to compare across the independent variables, is loan status. The strategy is to figure out compare the average default rates across various independent variables and identify the  ones that affect default rate the most.

# ### Data Cleaning

# In[190]:


loan.isnull().sum()


# In[191]:


round(100*(loan.isnull().sum()/len(loan.index)),2)


# In[192]:


missing_col=loan.columns[100*(loan.isnull().sum()/len(loan.index))>90]
missing_col


# In[193]:


loan=loan.drop(missing_col,axis=1)


# In[194]:


print(loan.shape)


# In[195]:


round(100*(loan.isnull().sum()/len(loan.index)),2)


# In[196]:


loan.loc[:,['desc','mths_since_last_delinq']].head()


# The column description contains the comments the applicant had written while applying for the loan. Although one can use some text analysis techniques to derive new features from this column (such as sentiment, number of positive/negative words etc.), we will not use this column in this analysis. 
# 
# Secondly, months since last delinquent represents the number months passed since the person last fell into the 90 DPD group. There is an important reason we shouldn't use this column in analysis - since at the time of loan application, we will not have this data , it cannot be used as a predictor of default at the time of loan approval. 
# 
# Thus drop the two columns.

# In[197]:


loan=loan.drop(['desc','mths_since_last_delinq'],axis=1)


# In[198]:


round(100*(loan.isnull().sum()/len(loan.index)),2)


# some columns are missing values,let check whether they are missing more values

# In[199]:


loan.isnull().sum(axis=1)


# In[200]:


len(loan[loan.isnull().sum(axis=1)>5].index)


# the data looks clean.lets look the all columns format

# In[201]:


loan.info()


# In[202]:


loan['int_rate']=loan['int_rate'].apply(lambda x:pd.to_numeric(x.split("%")[0]))


# In[203]:


loan=loan[~loan['emp_length'].isnull()]


# In[204]:


import re


# In[205]:


loan['emp_length']=loan['emp_length'].apply(lambda x:re.findall('\d+',str(x))[0])
loan['emp_length']=loan['emp_length'].apply(lambda x:pd.to_numeric(x))


# ### Data Analysis
# 
# Let's now move to data analysis. To start with, let's understand the objective of the analysis clearly and identify the variables that we want to consider for analysis. 
# 
# The objective is to identify predictors of default so that at the time of loan application, we can use those variables for approval/rejection of the loan. Now, there are broadly three types of variables - 1. those which are related to the applicant (demographic variables such as age, occupation, employment details etc.), 2. loan characteristics (amount of loan, interest rate, purpose of loan etc.) and 3. Customer behaviour variables (those which are generated after the loan is approved such as delinquent 2 years, revolving balance, next payment date etc.).
# 
# Now, the customer behaviour variables are not available at the time of loan application, and thus they cannot be used as predictors for credit approval. 
# 
# Thus, going forward, we will use only the other two types of variables.
# 

# In[206]:


behaviour_var =  [
  "delinq_2yrs",
  "earliest_cr_line",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "out_prncp",
  "out_prncp_inv",
  "total_pymnt",
  "total_pymnt_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_pymnt_d",
  "last_pymnt_amnt",
  "last_credit_pull_d",
  "application_type"]


# In[207]:


df=loan.drop(behaviour_var,axis=1)


# In[208]:


df.info()


# In[209]:


df = df.drop(['title', 'url', 'zip_code', 'addr_state'], axis=1)


# Next, let's have a look at the target variable - loan_status. We need to relabel the values to a binary form - 0 or 1, 1 indicating that the person has defaulted and 0 otherwise.

# In[210]:


df['loan_status']=df['loan_status'].astype('category')
df['loan_status'].value_counts()


# In[211]:


df=df[df['loan_status'] !='Current']
df['loan_status']=df['loan_status'].apply(lambda x: 0 if x=='Fully Paid' else 1)
df['loan_status']=df['loan_status'].apply(lambda x: pd.to_numeric(x))
df['loan_status'].value_counts()


# Next, let's start with univariate analysis
# 
# ### Univariate Analysis
# 
# First, let's look at the overall default rate.

# In[212]:


round(np.mean(df['loan_status']), 2)


# Let's first visualise the average default rates across categorical variables.
# 

# In[213]:


sns.barplot(x='grade',y='loan_status',data=df)
plt.show()


# In[214]:


def plot_var(cat_var):
    sns.barplot(x=cat_var,y='loan_status',data=df,orient='v')
    plt.show()


# In[215]:


plot_var('grade')


# Clearly, as the grade of loan goes from A to G, the default rate increases. This is expected because the grade is decided by Lending Club based on the riskiness of the loan. 

# In[216]:


plot_var('term')


# In[217]:


plt.figure(figsize=(16, 6))
plot_var('sub_grade')


# In[218]:


plot_var('home_ownership')


# In[219]:


plot_var('verification_status')


# In[220]:


plt.figure(figsize=(16, 6))
sns.countplot(x='purpose', data=df)
plt.show()


# In[221]:


df['issue_d'].head()


# In[222]:


from datetime import datetime
df['issue_d']=df['issue_d'].apply(lambda x: datetime.strptime(x,'%b-%y'))


# In[223]:


df['month']=df['issue_d'].apply(lambda x:x.month)
df['year']=df['issue_d'].apply(lambda x:x.year)


# In[224]:


df.groupby('year').year.count()


# In[225]:


df.groupby('month').month.count()


# In[226]:


plot_var('year')


# In[227]:


plt.figure(figsize=(16, 6))
plot_var('month')


# Let's now analyse how the default rate varies across continuous variables.

# In[228]:


sns.distplot(df['loan_amnt'])
plt.show()


# he easiest way to analyse how default rates vary across continous variables is to bin the variables into discrete categories.
# 
# Let's bin the loan amount variable into small, medium, high, very high.

# In[229]:


def loan_amt(n):
    if n<5000:
        return "low"
    elif n>=5000 and n<=15000:
        return 'medium'
    elif n>=15000 and n<=25000:
        return 'high'
    else :
        return 'very high'
    
df['loan_amnt']=df['loan_amnt'].apply(lambda x:loan_amt(x))


# In[230]:


df['loan_amnt'].value_counts()


# In[231]:


plot_var('loan_amnt')


# In[232]:


df['funded_amnt_inv']=df['funded_amnt_inv'].apply(lambda x: loan_amt(x))


# In[233]:


plot_var('funded_amnt_inv')


# In[234]:


def int_rate(n):
    if n<=10:
        return 'low'
    elif n>=10 and n<=15:
        return 'medium'
    else :
        return 'high'
    
df['int_rate']=df['int_rate'].apply(lambda x: int_rate(x))


# In[235]:


plot_var('int_rate')


# In[236]:


def dti(n):
    if n <= 10:
        return 'low'
    elif n > 10 and n <=20:
        return 'medium'
    else:
        return 'high'
    

df['dti'] = df['dti'].apply(lambda x: dti(x))


# In[237]:


plot_var('dti')


# In[238]:


def funded_amount(n):
    if n <= 5000:
        return 'low'
    elif n > 5000 and n <=15000:
        return 'medium'
    else:
        return 'high'
    
df['funded_amnt'] = df['funded_amnt'].apply(lambda x: funded_amount(x))


# In[239]:


plot_var('funded_amnt')


# In[240]:


def installment(n):
    if n <= 200:
        return 'low'
    elif n > 200 and n <=400:
        return 'medium'
    elif n > 400 and n <=600:
        return 'high'
    else:
        return 'very high'
    
df['installment'] = df['installment'].apply(lambda x: installment(x))


# In[241]:


plot_var('installment')


# In[242]:


def annual_income(n):
    if n <= 50000:
        return 'low'
    elif n > 50000 and n <=100000:
        return 'medium'
    elif n > 100000 and n <=150000:
        return 'high'
    else:
        return 'very high'

df['annual_inc'] = df['annual_inc'].apply(lambda x: annual_income(x))


# In[243]:


plot_var('annual_inc')


# In[244]:


df = df[~df['emp_length'].isnull()]

# binning the variable
def emp_length(n):
    if n <= 1:
        return 'fresher'
    elif n > 1 and n <=3:
        return 'junior'
    elif n > 3 and n <=7:
        return 'senior'
    else:
        return 'expert'

df['emp_length'] = df['emp_length'].apply(lambda x: emp_length(x))


# In[245]:


plot_var('emp_length')


# ### Segmented Univariate Analysis
# 
# We have now compared the default rates across various variables, and some of the important predictors are purpose of the loan, interest rate, annual income, grade etc.
# 
# In the credit industry, one of the most important factors affecting default is the purpose of the loan - home loans perform differently than credit cards, credit cards are very different from debt condolidation loans etc. 
# 
# This comes from business understanding, though let's again have a look at the default rates across the purpose of the loan.

# In[246]:


plt.figure(figsize=(16, 6))
plot_var('purpose')


# In[247]:


plt.figure(figsize=(16, 6))
sns.countplot(x='purpose', data=df)
plt.show()


# Let's analyse the top 4 types of loans based on purpose: consolidation, credit card, home improvement and major purchase.

# In[248]:


main_purpose= ["credit_card","debt_consolidation","home_improvement","major_purchase"]
df=df[df['purpose'].isin(main_purpose)]
df['purpose'].value_counts()


# In[249]:


sns.countplot(x=df['purpose'])
plt.show


# In[250]:


plt.figure(figsize=[10, 6])
sns.barplot(x='term', y="loan_status", hue='purpose', data=df)
plt.show()


# In[251]:


def plot_segmented(cat_var):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=cat_var, y='loan_status', hue='purpose', data=df)
    plt.show()

    
plot_segmented('term')


# In[252]:


plot_segmented('grade')


# In[253]:


plot_segmented('home_ownership')


# In[254]:


plot_segmented('year')


# In[255]:


plot_segmented('emp_length')


# In[256]:


plot_segmented('loan_amnt')


# In[257]:


plot_segmented('int_rate')


# In[258]:


plot_segmented('installment')


# In[259]:


plot_segmented('dti')


# In[260]:


plot_segmented('annual_inc')


# A good way to quantify th effect of a categorical variable on default rate is to see 'how much does the default rate vary across the categories'. 
# 
# Let's see an example using annual_inc as the categorical variable.

# In[261]:


df.groupby('annual_inc').loan_status.mean().sort_values(ascending=False)


# In[262]:


def diff_rate(cat_var):
    default_rates = df.groupby(cat_var).loan_status.mean().sort_values(ascending=False)
    return (round(default_rates, 2), round(default_rates[0] - default_rates[-1], 2))

default_rates, diff = diff_rate('annual_inc')
print(default_rates) 
print(diff)


# Thus, there is a 6% increase in default rate as you go from high to low annual income. We can compute this difference for all the variables and roughly identify the ones that affect default rate the most.

# In[263]:


df_categorical = df.loc[:, df.dtypes == object]
df_categorical['loan_status'] = df['loan_status']

print([i for i in df.columns])


# In[264]:


d = {key: diff_rate(key)[1]*100 for key in df_categorical.columns if key != 'loan_status'}
print(d)


# In[ ]:




