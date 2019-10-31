import pickle
import numpy as np 
import pandas as pd 

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor


if __name__ == '__main__':
    #storage 
    data_file = 'isolation_forest_training_data.csv'          #change this file name to your training data file
    model_file = "model.sav"        
    scaler_file = "scaler.sav"

    #classification results from training data
    normal_count = 0
    anomaly_count = 0

    #get data
    df = pd.read_csv(data_file)  

    #remove the time component
    df = df.drop('Time', axis=1)

    #drop missing data
    df = df.dropna()

    #convert type for sklearn
    training_data = np.asarray(df)

    #standardize 
    scaler = MinMaxScaler().fit(training_data)
    training_data_transformed = scaler.transform(training_data)
 
    #select your machine learning model. 
    #Remove the # from the model you choose and add a # to the one you're not using
    #Update the contamination parameter to the percent of anomalies in your training data
    model = IsolationForest(n_estimators=100, contamination=0.2, bootstrap=False)   # option 1
    #model = LocalOutlierFactor(novelty=True, contamination=0.1)                    # option 2 - novelty must be set to True!

    #train model
    model.fit(training_data_transformed)

    #get classification results 
    prediction = model.predict(training_data_transformed)
    for p in prediction:
        if p == 1:            
            normal_count += 1
        elif p == -1:
            anomaly_count += 1    
    print("normal\n\tcount: ", normal_count)
    print("anomaly\n\tcount: ", anomaly_count)

    #save model & scaler for later application
    pickle.dump(model, open(model_file, 'wb'))
    pickle.dump(scaler, open(scaler_file, 'wb'))
  

