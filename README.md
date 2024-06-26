# Design and deploy a credit scoring model


## Issues addressed by the project

The project involves implementing a **credit scoring** tool based on **Machine Learning** technologies and customer's informations to calculate the probability that he/she will be able to repay the credit they are applying for and help the company to decide to grant or refuse the application. In addition, this project must also meet a growing demand from customers for transparency in credit granting decisions.
Machine Learning has the reputation of being not very transparent and rather considered as a black box which does not favor its acceptance either.

>Nevertheless, enormous progress has been made to date and the field of Machine Learning offers very interesting solutions to improve the understanding of decision-making processes.


## Objectives

1. This tool should make it possible to define the probability of defaulting on a loan on the basis of information relating to the customer.
2. It must also offer a certain level of transparency and simplicity concerning the data and its processing, with a view to implementing methods for interpreting the model in the form of an interactive dashboard for use by customer-relations managers and their customers.
3. A simulation of the data drift is also carried out for the purpose of determining a maintenance period to maintain optimum prediction performances.


## The tool to develop

**Credit Scoring** is a risk analysis tool for granting credit. Based on statistical methods, it takes into account a lot of information relating to the loan applicant to assess the risk of non-repayment. In concrete terms, the tool assigns a score to a loan application by statistical analysis on a reference basis (expired files whose outcome is known).

Although very widespread in credit organizations, many aspects remain to be improved. Mainly:

**- trust**

The tool's impartiality is often cited as a virtue by the industry but does not necessarily convince applicants, especially when their credit application is refused.

**- transparency of information**

The new European regulation on the protection of personal data entered into force on May 25, 2018 (https://www.cnil.fr/fr/reglement-europeen-protection-donnees). Article 12 specifically deals with "Transparency of information and communications and modalities for exercising the rights of the data subject".

**- performances**

Regardless of the method used to establish credit application scores, the results can be misleading. This may be the case for "non-standard" files, for example, which encounter few similar cases in the processing sample.

>The quality of the reference data is therefore very important. The aim is to have a database that is as representative as possible of the situations to be dealt with.

The techniques and methods used to build the model obviously have a significant impact on its performance. No model is perfect and there will always be instances of poor predictions.

>The technical challenge is to minimize these cases.


## Project architecture

The project have been splitted into two parts:

### Part 1: Development

The first part of the project consists of realizing:
- an exploratory data analysis.
- their processing in order to build the dataframes adapted to the modeling of a Machine Learning tool.
- the modeling of a supervised learning system.
- the interpretation of the importance of features at local and global level.
- simulation of data drift to determine a maintenance period in order to maintain optimal prediction performance.
- the development of the web application which is divided into 2 sub-parts: one frontend based on streamlit (stored on GitHub: https://github.com/mpacaud/Design-and-deploy-a-credit-scoring-model--Frontend.git) and the other backend based on Flask (stored on GitHub: https://github.com/mpacaud/Design-and-deploy-a-credit-scoring-model--Backend.git).

NB: The fifth first points are realized in successive notebooks noted from N1 to N5.

### Part 2: Deployment

**Local**

To launch the application locally, just run:
- first the test server provided with Flask.
- then, the streamlit server which communicates to the previous server the requests sent by the user.

**On line**

The deployment of the script is carried out via the Heroku platform. The application dashboard can be loaded with the following link: https://mpacaud-oc-ds-p7-app-frontend.herokuapp.com

*NB1: By clicking on this link, it is possible to log into the backend server: https://mpacaud-oc-ds-p7-app-backend.herokuapp.com*

*NB2: It seems that the Flask application cannot work on Azure because a package is missing on their linux installation to run the chosen model (LightGBM).*


## Folder architecture of the repository

### Root

#### Files

- app.py: Flask API (backend part of the application).
- dashboard_streamlit: Interactive dashboard (frontend part of the application) configured to look for the Flask API on lan.
- All notebooks: Each development part.
- Requirements.txt: Environment packages used.
- test_\*: Unitary tests of both parts of the application (front and backend).
- urls: on-lan and on-line urls of the Flask API.

#### Folders

- mlruns: Centralized experiments' results and models.
- Exports:
  - Data drift: All concerning the data drift anaylisis with Evidently report table.
  - Feature_interpretation: All concerning model interpretability with SHAP.
  - Figures: Exported figures from the different notebooks.
  - Models: All models tried with different methods and hyperparameters optimization history.
  - Preprocessed_data: Data used by models adter their preprocessing within the notebook (N2) feature engineering.


## Further notes

For reasons of server resource management, it is possible that the application is not permanently maintained on the Heroku hosting site.
In order to remain under the 100 Mo file size limitation of GitHub the following minor files have not been uploaded:
  - mlruns
  - "\Exports\Feature_interpretation\SHAP\global_shap_explanations.pkl"
  - "\Exports\Models\Tried" (only models trained on undersampled datasets)
  - "\Exports\Preprocessed_data"