import numpy as np
import pandas as pd

def preprocess_data():
    patient_data = pd.read_csv("patient_data.csv", header=None) #[15001 x 10]
    medical_predictor_labels = [""]*(len(patient_data[0][0]) +1) # len = 10 , ['PatientID', 'Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age', 'Diabetic']
    
    i = 0
    for label in patient_data.iloc[0, :]:
        medical_predictor_labels[i] = label
        i += 1

    #extract patient dataset feature columns without headings
    patientID = np.array(patient_data.iloc[1:, 0])
    patientID = np.array(patient_data.iloc[1:, 1])
    plasmaGlucose = np.array(patient_data.iloc[1:, 2])
    diastolicBloodPressure = np.array(patient_data.iloc[1:, 3])
    tricepsThickness = np.array(patient_data.iloc[1:, 4])
    SerumInsulin = np.array(patient_data.iloc[1:, 5])
    bmi = np.array(patient_data.iloc[1:, 6])
    diabetesPedigree = np.array(patient_data.iloc[1:, 7])
    age = np.array(patient_data.iloc[1:, 8])
    diabetic = np.array(patient_data.iloc[1:, 9])

    #combine vectors to matrix
    temp = np.column_stack((patientID, patientID, plasmaGlucose, diastolicBloodPressure, tricepsThickness, SerumInsulin, bmi, diabetesPedigree, age))
    # append array of ones to temp with length = to amt of temp rows and 1 col
    oneVector = np.ones((temp.shape[0], 1))
    # combine ones vector with score vectors to make feature matrix
    feature_matrix_X = np.hstack((oneVector, temp)) #(15000, 10)
    diabetic_label_y = np.array(diabetic) # (15000)

    # randomly grabbing 90% of training set
    size90 = int(0.9 * feature_matrix_X.shape[0])
    #getting random training sample for y by selecting indices at random w/o replacement
    y_train_index = np.random.choice(np.arange(diabetic_label_y.size), size90, replace=False)
    y_train = [diabetic_label_y[i] for i in y_train_index]
    # #getting test sample for y by getting indices not chosen above
    y_test_index = np.setdiff1d(np.arange(diabetic_label_y.size), y_train_index)
    y_test = [diabetic_label_y[i] for i in y_test_index]
    # #getting random sample from x of training examples
    x_train_index = np.random.choice(feature_matrix_X.shape[0], size90, replace=False)
    x_train_index = x_train_index.astype(int) # convert to int
    x_train = feature_matrix_X[x_train_index] 
    #getting other 10%
    x_test_index = np.setdiff1d(np.arange(feature_matrix_X.shape[0]), x_train_index)
    x_test = feature_matrix_X[x_test_index]

    return [x_train, y_train, x_test, y_test]
    
        



def main():
    preprocess_data()


if __name__ == '__main__':
    main()


