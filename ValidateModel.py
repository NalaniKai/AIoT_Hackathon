import pickle
import numpy as np 
import pandas as pd 

'''
get_data    placeholder for collecting event data from Modi device
'''
def get_data(data_file):  
    #get data
    df = pd.read_csv(data_file)  

    #remove the time component
    df = df.drop('Time', axis=1)

    #drop missing data
    df = df.dropna()

    #convert type for sklearn
    return np.asarray(df)


'''
classify    Classifies data as anomaly or normal behavior 

@param model    trained model to classify event data
@param scaler   MinMaxScaler used to scale training data for the model
@param ANOMALY  Constant value signaling anomaly from model prediction 
'''
def classify(model, scaler, ANOMALY, data_file):
    #classification results from training data
    normal_count = 0
    anomaly_count = 0

    events = get_data(data_file)
    transformed_data = scaler.transform(events)
    classification = model.predict(transformed_data)

    for p in classification:
        if p == 1:            
            normal_count += 1
        elif p == -1:
            anomaly_count += 1    
    print("normal\n\tcount: ", normal_count)
    print("anomaly\n\tcount: ", anomaly_count)

    '''
    #print event data with labels. Delete the comment block to see data classifications.
    i = 0 
    for p in classification:
        if p == ANOMALY:
            print("Anomaly: ", events[i]) 
        else:
            print("Normal: ", events[i]) 
        i += 1
    '''

if __name__ == '__main__':
    ANOMALY = -1
    data_file = 'isolation_forest_normal_test_data.csv'
    model_file = "model.sav"
    scaler_file = "scaler.sav"

    #load model & scaler
    model = pickle.load(open(model_file, 'rb'))
    scaler = pickle.load(open(scaler_file, 'rb'))

    classify(model, scaler, ANOMALY, data_file)
