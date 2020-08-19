import numpy as np
import pandas as pd
from sklearn import preprocessing
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))
model5 = pickle.load(open('model5.pkl', 'rb'))
model6 = pickle.load(open('model6.pkl', 'rb'))
model7 = pickle.load(open('model7.pkl', 'rb'))
model8 = pickle.load(open('model8.pkl', 'rb'))
model9 = pickle.load(open('model9.pkl', 'rb'))
model10 = pickle.load(open('model10.pkl', 'rb'))
model11 = pickle.load(open('model11.pkl', 'rb'))
model12= pickle.load(open('model12.pkl', 'rb'))
model13 = pickle.load(open('model13.pkl', 'rb'))
model14 = pickle.load(open('model14.pkl', 'rb'))
#model15 = pickle.load(open('model15.pkl', 'rb'))
#model16 = pickle.load(open('model16.pkl', 'rb'))
#model17 = pickle.load(open('model17.pkl', 'rb'))
#model18 = pickle.load(open('model18.pkl', 'rb'))
#model19 = pickle.load(open('model19.pkl', 'rb'))
#model20 = pickle.load(open('model20.pkl', 'rb'))
#model21 = pickle.load(open('model21.pkl', 'rb'))




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
################################################################################

    features = [float(x) for x in request.form.values()]

################################################################################
    test = features
    final_features = [np.array(features)]
    prediction1 = model1.predict(final_features)
    test.append(prediction1)

    final_features = [np.array(test)]
    prediction2 = model2.predict(final_features)
    test.append(prediction2)

    final_features = [np.array(test)]
    prediction3 = model3.predict(final_features)
    test.append(prediction3)

    final_features = [np.array(test)]
    prediction4 = model4.predict(final_features)
    test.append(prediction4)

    final_features = [np.array(test)]
    prediction5 = model5.predict(final_features)
    test.append(prediction5)

    final_features = [np.array(test)]
    prediction6 = model6.predict(final_features)
    test.append(prediction6)

    final_features = [np.array(test)]
    prediction7 = model7.predict(final_features)
    test.append(prediction7)

    final_features = [np.array(test)]
    prediction8 = model8.predict(final_features)
    test.append(prediction8)

    final_features = [np.array(test)]
    prediction9 = model9.predict(final_features)
    test.append(prediction9)

    final_features = [np.array(test)]
    prediction10 = model10.predict(final_features)
    test.append(prediction10)

    final_features = [np.array(test)]
    prediction11 = model11.predict(final_features)
    test.append(prediction11)

    final_features = [np.array(test)]
    prediction12 = model12.predict(final_features)
    test.append(prediction12)

    final_features = [np.array(test)]
    prediction13 = model13.predict(final_features)
    test.append(prediction13)

    final_features = [np.array(test)]
    prediction14 = model14.predict(final_features)
    test.append(prediction14)

    final_features = [np.array(test)]
    #prediction15 = model15.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction16 = model16.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction17 = model17.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction18 = model18.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction19 = model19.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction20 = model20.predict(final_features)
    test.append(0)

    final_features = [np.array(test)]
    #prediction21 = model21.predict(final_features)
    test.append(0)


    output1 = round(prediction1[0], 2)
    output2 = round(prediction2[0], 2)
    output3 = round(prediction3[0], 2)
    output4 = round(prediction4[0], 2)
    output5 = round(prediction5[0], 2)
    output6 = round(prediction6[0], 2)
    output7 = round(prediction7[0], 2)
    output8 = round(prediction8[0], 2)
    output9 = round(prediction9[0], 2)
    output10 = round(prediction10[0], 2)
    output11 = round(prediction11[0], 2)
    output12 = round(prediction12[0], 2)
    output13 = round(prediction13[0], 2)
    output14 = round(prediction14[0], 2)
    output15 = 0
    output16 = 0
    output17 = 0
    output18 = 0
    output19 = 0
    output20 = 0
    output21 = 0

    l=[]
    l.append(output1)
    l.append(output2)
    l.append(output3)
    l.append(output4)
    l.append(output5)
    l.append(output6)
    l.append(output7)
    l.append(output8)
    l.append(output9)
    l.append(output10)
    l.append(output11)
    l.append(output12)
    l.append(output13)
    l.append(output14)
    l.append(output15)
    l.append(output16)
    l.append(output17)
    l.append(output18)
    l.append(output19)
    l.append(output20)
    l.append(output21)

################################################################################################
    medicine = ['Acetohexamide','Glimepiride Pioglitazone', 'Metformin Pioglitazone', 
     'Metformin Rosiglitazone','Troglitazone', 'Glipizide Metformin','Tolbutamide','Miglitol','Tolazamide',
     'Chlorpropamide','Acarbose','Nateglinide','Glyburide Metformin','Repaglinide','Glimepiride', 
     'Rosiglitazone','Pioglitazone','Glyburide', 'Glipizide','Metformin','Insulin']
#################################################################################################

    df1=pd.read_csv("clean_diabetic.csv")
    
    # l has the predictions

    med = ['metformin',
    'repaglinide',  #13
    'nateglinide',  #11
    'chlorpropamide',  #9
    'glimepiride', #14
    'glipizide',  #18
    'glyburide',   #17
    'pioglitazone',   #16
    'rosiglitazone',  #15
    'acarbose',   #10
    'miglitol',  #7
    'insulin',  #19
    'glyburide_metformin',  #12
    'tolazamide',   #8
    'metformin_pioglitazone',  #2
    'metformin_rosiglitazone',  #3
    'glimepiride_pioglitazone', #1
    'glipizide_metformin',#5
    'troglitazone',  #4
    'tolbutamide','acetohexamide' #6
    ]

    df_new = pd.DataFrame() #containing the medicines
    for i in med:
        df_new[i] = df1[i]
    
    

    df = df1.drop(['metformin',
    'repaglinide',  #13
    'nateglinide',  #11
    'chlorpropamide',  #9
    'glimepiride', #14
    'glipizide',  #18
    'glyburide',   #17
    'pioglitazone',   #16
    'rosiglitazone',  #15
    'acarbose',   #10
    'miglitol',  #7
    'insulin',  #19
    'glyburide_metformin',  #12
    'tolazamide',   #8
    'metformin_pioglitazone',  #2
    'metformin_rosiglitazone',  #3
    'glimepiride_pioglitazone', #1
    'glipizide_metformin',#5
    'troglitazone',  #4
    'tolbutamide','acetohexamide' #6
    ], axis = 1)


    ####################
    
    # df.select_dtypes(include=['category'])
    categorical_features = [key for key in dict(df.dtypes)
                if dict(df.dtypes)[key] in ['object'] ] # Categorical Varible

    # print(cat_var)
    def encrypt_single_column(data,column):
        le = preprocessing.LabelEncoder()
        le.fit(data.astype(str))
        #le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        return le.transform(data.astype(str))

    def encrypt_columns_collection(data, columns_to_encrypt):
        for column in columns_to_encrypt:
            data[column] = encrypt_single_column(data[column],column)
        return data

    df = encrypt_columns_collection(df, categorical_features)

    flag=0
    count=0
    for i in range(len(df)):
        if(df['encounter_id'][i] == features[1] or df['patient_nbr'][i] == features[2]):
            flag=1
            count=i
            break
        




########################################################################################################





    if(flag==1):
        l = list(df_new.iloc[count])
        
        med = ['Metformin',
    'Repaglinide',  #13
    'Nateglinide',  #11
    'Chlorpropamide',  #9
    'Glimepiride', #14
    'Glipizide',  #18
    'Glyburide',   #17
    'Pioglitazone',   #16
    'Rosiglitazone',  #15
    'Acarbose',   #10
    'Miglitol',  #7
    'Insulin',  #19
    'Glyburide Metformin',  #12
    'Tolazamide',   #8
    'Metformin Pioglitazone',  #2
    'Metformin Rosiglitazone',  #3
    'Glimepiride Pioglitazone', #1
    'Glipizide Metformin',#5
    'Troglitazone',  #4
    'Tolbutamide','Acetohexamide' #6
    ]
    
        pred = []
        not_pred = []
        for i in range(len(l)):
            if(l[i]==1):
                pred.append(med[i])
            else:
                not_pred.append(med[i])

        
        if(pred == []):
            pred.append("No Medicine Prescribed")

        temp1="Medicines to be taken: "
        for i in pred:
            temp1 = temp1+str(i)+", \t"
        
        temp2 = "Medicines not to be taken: "
        for i in not_pred:
            temp2 = temp2+str(i)+", \t"

        
        
        return render_template('final.html', prediction_text1='{}'.format(temp1),prediction_text2='{}'.format(temp2))#change l

    
    else:
        pred = []
        not_pred = []
        for i in range(len(l)):
            if(l[i]==1):
                pred.append(medicine[i])
            else:
                not_pred.append(medicine[i])

        if(pred == []):
            pred.append("No Medicine Prescribed")

        temp1=""
        for i in pred:
            temp1 = temp1+str(i)+", \t"
        
        temp2 = ""
        for i in not_pred:
            temp2 = temp2+str(i)+", \t"

        temp = temp1+"\n"+temp2


    
        return render_template('final.html', prediction_text1='{}'.format(temp1),prediction_text2='{}'.format(temp2))#change l




if __name__ == "__main__":
    app.run(debug=True)