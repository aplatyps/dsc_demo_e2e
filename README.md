# Fraud detection with artificial neural network

A locally-hosted demo web app built with Python, FLASK and SQLite to determine whether a job post is legitimate or fraudulent. The dataset is trained on with artificial neural network. 

### Web page:

Self explanatory landing page

![Home_page1](figures/webapp_page1_1.PNG)
![Home_page2](figures/webapp_page1_2.PNG)
![Home_page3](figures/webapp_page1_3.PNG)
![Home_page4](figures/webapp_page1_4.PNG)

Information on the dataset

![db_page](figures/webapp_page2_1.PNG)

Job fraud detection page. For now the web app accepts input as a .csv file. The input will be fed to the saved model for deployment.

![jf_page1](figures/webapp_page3_1.PNG)

Output of the model is returned as a list for now. Not planning to do UI/UX here.

![jf_page2](figures/webapp_page3_2.PNG)


### Methodology

Dataset sample is too small with highly imbalanced class. Oversampling of the class with lower samples is used. Dataset is then splitted into training, validation and test set. Data is heavily pre-processed before use.

### Performance:

Accuracy plot

![Accuracy](figures/fraud_detection_mlp_accuracy.png) 

Loss plot

![Loss](figures/fraud_detection_mlp_loss.png)

Confusion matrix plot

![Confusion](figures/fraud_detection_mlp_confusion.png)

Report of test dataset results

![Result](figures/model_test_result.PNG)
