# Used Car Pricing (Fair Market Value) Prediction Webapp

This is a machine learning web application, which utilizes a CatBoost Regression algorithm trained on the [US Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) for the purpose of predicting a car's price given its features, with the predicted price being in USD.

The data was initially cleaned and various regression algorithms were evaluated based on their MAE, RMSE and R2 scores. The CatBoost algorithm performed the best without requiring any external preprocessing of the data. It was then tuned using a randomized hyperparameter search. The best model weights were then saved and used for this implementation. The weights can be downloaded from [here](https://github.com/mo-adi/fmv_webapp/releases/download/v1.0/catboost_model_tuned.sav).

## Requirements:
* Python - 3.8+
* NumPy - 1.23.4
* Pandas - 1.5.1
* Scikit-learn - 1.1.3
* Streamlit - 1.14.0
* Catboost - 1.0.6

## Install requirements:
```bash
pip install -r requirements.txt
```

## Run web application:
```bash
streamlit run app.py
```
