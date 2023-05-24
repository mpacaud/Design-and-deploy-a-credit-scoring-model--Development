#!/usr/bin/env python
# coding: utf-8

# # Projet 7 - Implementation of a scoring model
# # Notebook - Model comparison

# # Context
# 
# 
# 

# # Data sources
# 
# The webpage containing all data and descriptions: <a href="https://www.kaggle.com/c/home-credit-default-risk/data" target="_blank">here</a>.

# # Glossary

# __- TP:__ True positives correspond to customers which are classified as they would default the repayment of their loan and they would as expected.<br>
# __- FP:__ False positives correspond to customers which were guessed trustless to repay their loans whereas they would have to (Secondary case to avoid and minimize if possible).<br>
# __- FN:__ False negatives correspond to customers which were guessed trustful to repay their loans whereas they will not (Worst case to absolutly minimize).<br>
# __- TN:__ True negatives correspond to customers which are classified as they would not default the repayment of their loan and they don't as expected.<br>
# __- dt_sp:__ Data sampling.<br>
# __- wt:__ Weight.<br>
# __- opt:__ Optimal.<br>
# __- synth_sp:__ Synthetic sampling.<br>
# 

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Projet-7---Implementation-of-a-scoring-model" data-toc-modified-id="Projet-7---Implementation-of-a-scoring-model-1">Projet 7 - Implementation of a scoring model</a></span></li><li><span><a href="#Notebook---Model-comparison" data-toc-modified-id="Notebook---Model-comparison-2">Notebook - Model comparison</a></span></li><li><span><a href="#Context" data-toc-modified-id="Context-3">Context</a></span></li><li><span><a href="#Data-sources" data-toc-modified-id="Data-sources-4">Data sources</a></span></li><li><span><a href="#Glossary" data-toc-modified-id="Glossary-5">Glossary</a></span></li><li><span><a href="#I)-Importation-of-the-dataset-into-a-pandas-dataframe" data-toc-modified-id="I)-Importation-of-the-dataset-into-a-pandas-dataframe-6">I) Importation of the dataset into a pandas dataframe</a></span><ul class="toc-item"><li><span><a href="#1)-Import-all-librairies-and-tools-required-to-realize-the-project-and-set-the-first-global-variables" data-toc-modified-id="1)-Import-all-librairies-and-tools-required-to-realize-the-project-and-set-the-first-global-variables-6.1">1) Import all librairies and tools required to realize the project and set the first global variables</a></span></li><li><span><a href="#2)-Importation-of-the-preprocessed-datasets" data-toc-modified-id="2)-Importation-of-the-preprocessed-datasets-6.2">2) Importation of the preprocessed datasets</a></span></li><li><span><a href="#3)-Separation-of-the-explicatives-and-the-explicated" data-toc-modified-id="3)-Separation-of-the-explicatives-and-the-explicated-6.3">3) Separation of the explicatives and the explicated</a></span></li></ul></li><li><span><a href="#Interpretations" data-toc-modified-id="Interpretations-7">Interpretations</a></span><ul class="toc-item"><li><span><a href="#Global" data-toc-modified-id="Global-7.1">Global</a></span><ul class="toc-item"><li><span><a href="#LightGBM-importance-parameter" data-toc-modified-id="LightGBM-importance-parameter-7.1.1">LightGBM importance parameter</a></span></li><li><span><a href="#SHAP" data-toc-modified-id="SHAP-7.1.2">SHAP</a></span><ul class="toc-item"><li><span><a href="#Library-importation" data-toc-modified-id="Library-importation-7.1.2.1">Library importation</a></span></li><li><span><a href="#Functions" data-toc-modified-id="Functions-7.1.2.2">Functions</a></span></li><li><span><a href="#Shap-explanation" data-toc-modified-id="Shap-explanation-7.1.2.3">Shap explanation</a></span></li><li><span><a href="#Interpretations" data-toc-modified-id="Interpretations-7.1.2.4">Interpretations</a></span></li></ul></li></ul></li><li><span><a href="#Local" data-toc-modified-id="Local-7.2">Local</a></span><ul class="toc-item"><li><span><a href="#Shap" data-toc-modified-id="Shap-7.2.1">Shap</a></span><ul class="toc-item"><li><span><a href="#Shap-explanation" data-toc-modified-id="Shap-explanation-7.2.1.1">Shap explanation</a></span></li><li><span><a href="#Interpretations" data-toc-modified-id="Interpretations-7.2.1.2">Interpretations</a></span></li></ul></li><li><span><a href="#Dashboard:-streamlit" data-toc-modified-id="Dashboard:-streamlit-7.2.2">Dashboard: streamlit</a></span></li><li><span><a href="#LIME" data-toc-modified-id="LIME-7.2.3">LIME</a></span></li></ul></li></ul></li><li><span><a href="#API-Flask" data-toc-modified-id="API-Flask-8">API Flask</a></span></li></ul></div>

# # I) Importation of the dataset into a pandas dataframe

# ## 1) Import all librairies and tools required to realize the project and set the first global variables

# In[1]:

import sys
sys.path.insert(0, 'D:/0Partage/MP-P2PNet/MP-Sync/MP-Sync_Pro/Info/OC_DS/Projet 7/GitRepo/')
# Now the import will work
from shared_functions import *


# ## 2) Importation of the preprocessed datasets

# In[2]:


df_train = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_train.csv'))
df_valid = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_valid.csv'))
df_test = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_test.csv'))
#df_new_customers = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_new_customers.csv'))


# In[3]:


df_test_w_ids = df_test.copy()


# In[4]:


del_features = ['SK_ID_CURR'] #, 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index', 'level_0']

df_train = df_train.drop(del_features, axis=1)
df_valid = df_valid.drop(del_features, axis=1)
df_test = df_test.drop(del_features, axis=1)


# In[5]:


get_ipython().run_cell_magic('time', '', '\ndf_train = find_int_cols(df_train)\ndf_train.info()\n')


# In[6]:


df_train = reduce_memory(df_train)


# In[7]:


df_train.info()


# ## 3) Separation of the explicatives and the explicated

# In[8]:


X_TRAIN = df_train.drop('TARGET', axis=1)
y_TRAIN = df_train['TARGET']

#X_VALID = df_valid.drop('TARGET', axis=1)
#y_VALID = df_valid['TARGET']

X_TEST = df_test.drop('TARGET', axis=1)
y_TEST = df_test['TARGET']


# In[9]:


model_pl_label = 'wt_lgbm_clf_fine_opt'

# Load the last values calculated for the hyperparameters.
df_MODELS = pd.read_pickle(os.path.join(EXPORTS_MODELS_DIR_PATH, PKL_MODELS_FILE))#.set_index('Model_labels')
model_pl_opt = df_MODELS.loc[model_pl_label, 'Models']


# In[10]:


model_pl_opt


# # Interpretations

# ## Global

# ### LightGBM importance parameter

# In[11]:


n_top_fts = 30

model_pl_opt['model'].fit(X_TRAIN, y_TRAIN)

npa_top_fts = model_pl_opt['model'].feature_importances_.argsort()#[-n_top_fts:]

plt.figure(figsize=(6,9), dpi=300)

plt.barh(X_TRAIN.columns[npa_top_fts[-n_top_fts:]], model_pl_opt['model'].\
         feature_importances_[npa_top_fts[-n_top_fts:]] / len(X_TRAIN.columns) * 100)

plt.xlabel("Feature Importance (%)")
plt.title("Top %i most important features" % n_top_fts)

plt.show()


# 

# ### SHAP

# - doc 1: https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
# - doc 2: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.dependence_plot.html
# 
# *NB: More about Kernel Explainer and its requirements for big datasets that Tree explainer bypasses:* https://colab.research.google.com/drive/1pjPzsw_uZew-Zcz646JTkRDhF2GkPk0N#scrollTo=SCOJFGMj3fq5

# #### Library importation

# In[12]:


import shap

shap.initjs()


# #### Functions

# #### Shap explanation

# - Different explainers: https://snyk.io/advisor/python/shap/functions/shap.explainers.explainer.Explainer
# - Background data or not & feature_perturbation = "interventional" or "tree_path_dependent" ? https://github.com/slundberg/shap/issues/1098 <br>
# => Background data: Closer to the model. <br>
# 
# *NB: According to the model (classifier or regressor) and the presence or not of background data, some graphics (such as shap.plots.bar()) won't behave the same way and might not be usable (for the classifiers mainly). Ex: shap.plots.bar can replaced by shap.plot_bar but such graphics are less detailed (as it can be noticed in a couple of cells below).*

# In[13]:


get_ipython().run_cell_magic('time', '', '\n# Select the number of the most important features to keep.\ntop_ft = 10\n\n# Select the categorical class to base the explanations on (ex: 0 or 1 for binaries).\ncat_class = 0\n\n# Get the pipeline to use.\nmodel_pl_label = \'wt_lgbm_clf_fine_opt\'\nmodel_pl = df_MODELS.loc[model_pl_label][\'Models\']\n\n# Get the model and the scaler of the pipeline.\nscaler = model_pl[\'scaler\']\nclf = model_pl[\'model\']\n\n# Scale the train and the test set to fit the model pipeline inputs format.\nX_train_norm = scaler.transform(X_TRAIN)\n#X_test_norm = scaler.transform(X_TEST)\nX_test_norm = pd.DataFrame(scaler.transform(X_TEST), columns=X_TEST.columns)\n#X_test_norm_mean = pd.DataFrame(X_test_norm.mean(axis=0).reshape(1, X_TEST.shape[1]), columns=X_TEST.columns)\n\n\n# Get a sample of the scaled train set to accelerate the training process.\n# 1. Sampling for KernelExplainer():\n#X_train_norm_sp = shap.sample(X_train_norm, 100, random_state=None) # use 600 samples of train data as background data\n#X_test_norm_sp = shap.sample(X_test_norm, 5, random_state=None)\n\nt0 = time()\n\n# Create the explainer.\n# 1. Config for KernelExplainer:\n#explainer= shap.KernelExplainer(clf.predict_proba, X_train_norm_sp)\n\n# 2. Config for TreeExplainer():\nexplainer_shap = shap.TreeExplainer(clf, X_train_norm, model_output="probability")\n\n# Get the shapley values.\n# 1. Config for KernelExplainer:\n#explanations = explainer_shap.explanations(X_test_norm_sp, l1_reg=\'aic\')\n\n# 2. Config for TreeExplainer:\n# NB: Get shap values only: explanations = explainer_shap.explanations(X_test_norm)\nexplanations = explainer_shap(X_test_norm, check_additivity=False)\n\ndelta_t = time() - t0\n\n# Transform log odd values to odd (easier to understand for clients and staffs).\nyhat = clf.predict_proba(X_test_norm)\nexplanations_transformed = logodd_to_odd(explanations, yhat, cat_class)\n\n\n# Summarize relevant values.\nprint("Model used:\\n", clf)\nprint()\nprint("Model mean prediction probabilities on test data:", yhat.mean(axis=0))\nprint()\nN_proba_mean = 1 - clf.predict(X_train_norm).mean(axis=0)\nprint("Model mean proportions of negative predictions (=> Ratio of accepted applications):", N_proba_mean)\nN_shap_proba_mean = explanations_transformed.base_values.mean(axis=0)\nprint("SHAP expected mean proportions of negative predictions:", N_shap_proba_mean) #explainer_shap.expected_value\nprint("SHAP explanation reliability index =", 1 - abs(N_shap_proba_mean - N_proba_mean))\nprint()\nprint("SHAP explainer run time:", round(delta_t, 2), "s")\nprint()\n')


# #### Interpretations

# __- Mean absolute shapley values__

# In[14]:


# Method 1:
shap.summary_plot(explanations, max_display=top_ft, plot_type='bar') #feature_names=X_TEST.columns,


# In[15]:


# Method 2 (with the absolute mean shapley values displayed):
# NB: As no feature_names parameter is present X_test_norm are reassociated with them in a df.
shap.plots.bar(explanations_transformed, max_display=top_ft+1)


# In[16]:


# Method 3 (with the shapley values as text):
# Source: https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137

from scipy.special import softmax

def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")
        
print_feature_importances_shap_values(explanations, X_TEST.columns)


# In[17]:


# Method 4 (Beeswarm):
# NB: Equivalent plot: shap.summary_plot(explanations, X_test_norm, feature_names=X_TEST.columns, max_display=top_ft, plot_type='dot')
shap.plots.beeswarm(explanations, max_display=top_ft+1)


# __- Features' influences and dependences__

# In[ ]:


# Detail features influences on predictions.
shap.dependence_plot('EXT_SOURCE_3', explanations.values, X_test_norm, interaction_index='EXT_SOURCES_MEAN') #feature_names=X_TEST.columns


# ## Local

# ### Shap

# #### Shap explanation

# In[19]:


# Select the number of the most important features to keep.
top_ft = 10

# Select the client.
client_idx = 0

# Select the categorical class to base the explanations on (ex: 0 or 1 for binaries).
cat_class = 0

# Get the pipeline to use.
model_pl_label = 'wt_lgbm_clf_opt'
model_pl = df_MODELS.loc[model_pl_label]['Models']

# Get the model and the scaler of the pipeline.
scaler = model_pl['scaler']
clf = model_pl['model']

# Scale the train and the test set to fit the model pipeline inputs format.
X_train_norm = scaler.transform(X_TRAIN)
X_test_norm = scaler.transform(X_TEST) 

# Get a sample of the scaled train set to accelerate the training process.
# 1. Sampling for KernelExplainer():
#X_train_norm_sp = shap.sample(X_train_norm, 100, random_state=None) # use 600 samples of train data as background data
#X_test_norm_sp = shap.sample(X_test_norm, 5, random_state=None)

# 2. Sampling for TreeExplainer():
X_test_norm_sp = pd.DataFrame(X_test_norm[client_idx].reshape(1,-1), columns=X_TEST.columns)

t0 = time()

# Create the explainer model (simplified .
# 1. Config for KernelExplainer:
#explainer_shap = shap.KernelExplainer(clf.predict_proba, X_train_norm_sp)

# 2. Config for TreeExplainer():
explainer_shap = shap.TreeExplainer(clf, X_train_norm)#, X_train_norm, model_output='probability')


# Get explanations
#explanations = explainer_shap(pd.DataFrame(X_test_norm_sp, columns=X_TEST.columns))
explanations = explainer_shap(X_test_norm_sp)


# Get the shapley values.
# 1. Config for KernelExplainer:
#shap_values = explainer_shap.shap_values(X_test_norm_sp, l1_reg='aic')

# 2. Config for TreeExplainer:
# NB: Get shap values only: shap_values = explainer_shap.shap_values(X_test_norm)
shap_values = explainer_shap.shap_values(X_test_norm_sp) #explainer_shap.shap_values(X_test_norm_sp) #explainer_shap.shap_values(X_test_norm_sp)

delta_t = time() - t0

# Transform log odd values to odd (easier to understand for clients and staffs).
yhat = clf.predict_proba(X_test_norm)
explanations_transformed = logodd_to_odd(explanations, yhat, cat_class)


# Summarize relevant values.
print("Model used:\n", clf)
print()
print("Model mean prediction probabilities on test data:", yhat.mean(axis=0))
print()
N_proba_mean = 1 - clf.predict(X_train_norm).mean(axis=0)
print("Model mean proportions of negative predictions (=> Ratio of accepted applications):", N_proba_mean)
N_shap_proba_mean = explanations_transformed.base_values.mean(axis=0)
print("SHAP expected mean proportions of negative predictions:", N_shap_proba_mean) #explainer_shap.expected_value
print("SHAP explanation reliability index =", 1 - abs(N_shap_proba_mean - N_proba_mean))
print()
print("Model prediction probabilities for the tested data:", clf.predict_proba(X_test_norm_sp)[0])
print()
print("SHAP explainer run time:", round(delta_t, 2), "s")
print()


# #### Interpretations

# __- Mean absolute shapley values__

# In[20]:


# Method 1:
shap.summary_plot(explanations_transformed, max_display=top_ft, plot_type='bar') #feature_names=X_TEST.columns,


# In[21]:


# Method 2 (with the absolute mean shapley values displayed):
# NB: As no feature_names parameter is present X_test_norm are reassociated with them in a df.
shap.plots.bar(explanations_transformed, max_display=top_ft+1)


# In[22]:


# Method 3 (with the mean shapley values displayed):
# NB: As no feature_names parameter is present X_test_norm are reassociated with them in a df.
shap.plots.bar(explanations_transformed[0], max_display=top_ft+1)


# __- Force plots__

# In[23]:


### Shape 1: Condensed ###

# NB1: f(x) scale coorresponds to the log odd (=> It shoud be <= 0 for positive ).
#      => In order to get the corresponding probability P = 10^(log odd value).
# NB2: .force_plot() seems to be the same as .plots.force().
#shap.plots.force(explainer_shap.expected_value[1], explanations[1], X_test_norm_sp, feature_names=X_TEST.columns)
shap.plots.force(explanations_transformed, X_test_norm_sp, feature_names=X_TEST.columns)


# In[24]:


### Shape 2: Detailed ###

# Transform the log odds default model values to their odd counterpart which
# is an easier scale to interpret for none professional people.
# NB: The argument "show=False" allows to not display the graph immediately in order to allows further customization with
#     matplotlib or seaborn or other... before displaying the graph with a "plt.show()".
shap_values_transformed = logodd_to_odd(explanations, yhat, cat_class)
shap.plots.waterfall(explanations_transformed[cat_class], top_ft+1, show=False)  
plt.axvline(x=0.6, color='k', linestyle='--')
#plt.xticks()
plt.show()


# In[ ]:





# ### Dashboard: streamlit

# In[25]:


#from pathlib import Path
#import pandas as pd
import streamlit as st
#from streamlit_jupyter import StreamlitPatcher, tqdm

#import streamlit.components.v1 as components

#StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers





@st.cache
def load_data():
    df = pd.read_csv(os.path.join(IMPORTS_DIR_PATH, 'preprocessed_data_test.csv'))
    return df

def st_shap(shap_plot, height=10):
    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    components.html(shap_html, height=height)


def shap_plot_waterfall ():
    shap_values_transformed = logodd_to_odd(explanations, yhat, cat_class)
    fig = shap.plots.waterfall(explanations_transformed[cat_class], top_ft+1, show=False)  
    fig.axvline(x=0.6, color='k', linestyle='--')
    #plt.xticks()
    st.show(fig)
    #plt.show()
    
df = load_data()
    
selected_customer_id = st.selectbox("Customer ID", df['SK_ID_CURR'])#, index=0, format_func=special_internal_function, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")


col1, col2 = st.columns(2)

with col1:
    st.header("Gauge")
    st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)#, min_value=None, max_value=None, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")


#name = st.text_input("CouCou")

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#st_shap(shap.plots.bar(explanations_transformed, max_display=top_ft+1))
st.write(shap.plots.force(explanations_transformed, X_test_norm_sp, feature_names=X_TEST.columns))


col1, col2 = st.columns(2)


with col1:
    st.header("Absolute SHAP values")
    st.write(shap.plots.bar(explanations_transformed, max_display=top_ft+1))

with col2:
    st.header("Detailed force plot")
    st.write(shap_plot_waterfall())

    
st.dataframe(df[df['SK_ID_CURR'] == selected_customer_id])




from flask import Flask
#from .views import app


# In[ ]:


app = Flask(__name__)

@app.route('/api/flask/')
def index():
    return "Hello world !"

if __name__ == "__main__":
    app.run()


# In[ ]:


@app.route('/api/flask/')
def api_flask ():
    pass # ligne temporaire


# In[ ]:




