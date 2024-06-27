import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from sklearn.metrics import r2_score
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
import pickle
from lightgbm import LGBMRegressor
import logging
import argparse

def setup_custom_logger(name, log_file):
    """
    Set up a custom logger that writes to a specified log file.
    
    :param name: The name of the logger.
    :param log_file: The file to which logs will be written.
    :return: A configured logger instance.
    """
    # Create a custom logger
    logger = logging.getLogger(name)
    
    # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(logging.DEBUG)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    
    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    return logger

logger = setup_custom_logger('mylogger', 'my_log_file.log')

def load_data(path):
    """Load data from CSV file"""
    df = pd.read_csv(path,index_col=0,parse_dates=True)
    return df

def train_test_split(data,test_size=12):
    """Split the data into train and test"""
    train = data[:-test_size]
    test = data[-test_size:]
    return train,test 

def process_data(data):
    """Process data and convert it into format used by Statsforecast and Neuralforecast"""
    data = data.reset_index()
    data.columns = ['ds','y']
    data['unique_id'] = 'Alcohol_Sale'
    return data

def show_train_test(train,test):
    """Function to display plot after train test split"""
    plt.figure(figsize=(20,8))
    plt.plot(train.index,train['y'],label='Training Data')
    plt.plot(test.index,test['y'],label='Test Data')
    plt.show()

class Forecasting_Model:
    """Forecast Model class to train, predict on data using different models"""
    def __init__(self,model_name,train,test,**params):
        self.model_name = model_name
        self.params = params
        self.train = train
        self.test = test
    def fit(self):
        if self.model_name == 'NBEATS':
            models = [NBEATS(**self.params)]
            self.model = NeuralForecast(models=models, freq='M')
            self.model.fit(self.train)
        elif self.model_name == 'NHITS':
            models = [NHITS(**self.params)]
            self.model = NeuralForecast(models=models, freq='M')
            self.model.fit(self.train)
        elif self.model_name == 'AutoARIMA':
            models = [AutoARIMA(**self.params)]
            self.model = StatsForecast(models=models, freq='M')
            self.model.fit(self.train)
        elif self.model_name == 'LightGBM':
            self.model = LGBMRegressor(**self.params)
            self.model.fit(self.train.index.values.reshape(-1,1),self.train['y'].values)

    def forecast(self,horizon=12):
        if self.model_name in ['NBEATS','NHITS']:
            self.forecast_ = self.model.predict().reset_index()
            self.forecast_.columns = ['unique_id','ds','forecast']
            self.forecast_ = self.forecast_['forecast'].values
            return self.forecast_
        elif self.model_name == 'AutoARIMA':
            self.forecast_= self.model.predict(h=horizon).reset_index()
            self.forecast_.columns = ['unique_id','ds','forecast']
            self.forecast_ = self.forecast_['forecast'].values
            return self.forecast_
        elif self.model_name == 'LightGBM':
            self.forecast_ = self.model.predict(test.index.values.reshape(-1,1))
            return self.forecast_

    def wmape(self):
        y_true = self.test['y'].values
        y_pred = self.forecast_
        return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    
    def plot_forecast(self,show=False):
        plt.figure(figsize=(20,8))
        # plt.plot(self.train.index,self.train['y'],label='Training Data')
        plt.plot(self.test.index,self.test['y'],label='Test Data')
        plt.plot(self.test.index,self.forecast_,label='Forecast')
        if show:
            plt.show()
        else:
            plt.savefig('forecast_image.png')
            plt.close()
 

def run_experiment(model_name,train_data,test_data,**params):
    """Running and tracking MLFlow Experiement for each type of model"""
    logger.info("Experiment Started Successfully")
    mlflow.set_experiment(model_name)
    logger.info(f"Model Name : {model_name}")
    with mlflow.start_run():
        model = Forecasting_Model(model_name=model_name,train=train,test=test,**params)
        model.fit()
        logger.info(f"Model Fitted")
        pred = model.forecast(horizon=12)
        logger.info(f"Forecast Generated")
        wmape = model.wmape()
        logger.info(f"Error Calculated")
        model.plot_forecast()
        logger.info(f"Plot Generated")
        with open('model.pkl','wb') as file:
            pickle.dump(model,file)
        mlflow.log_metric("WMAPE", wmape)
        if wmape>0.2:
            logger.warning(f'Poor WMAPE : {wmape}')
        mlflow.log_params(params)
        mlflow.log_artifact('forecast_image.png')
        mlflow.log_artifact('model.pkl')
        logger.info(f"Experiment logged")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri")
    args = parser.parse_args()
    logger.info('New Execution---------------')

    #Starting MLFlow Session
    mlflow.set_tracking_uri(args.uri)
    data = load_data('/teamspace/uploads/Alcohol_Sales.csv')
    logger.info(f"Data Loaded")
    df_processed = process_data(data)
    logger.info(f"Data Processed")
    train,test = train_test_split(df_processed)
    logger.info(f"Train Test Split")

    # LightGBM
    run_experiment('LightGBM',train,test,n_estimators=1000,learning_rate=0.001)
    run_experiment('LightGBM',train,test,n_estimators=500,learning_rate=0.001)
    
    # AutoARIMA
    for season_length in [6,12,18,24]:
        run_experiment('AutoARIMA',train,test,season_length=season_length)
    
    # NBEATS
    for input_size in [6,12,18,24]:
        for max_steps in [50,100,150]:
            run_experiment('NBEATS',train,test,input_size=input_size, h=12, max_steps=max_steps)