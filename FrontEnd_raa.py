# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:50:17 2023

@author: dell
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from PIL import Image

#loading model

road_accident_model=pickle.load(open('C:/Users/dell/OneDrive/Desktop/Model/Saved Model/road_accident_model.sav','rb'))

#sidebar for navigation
df=pd.read_csv('D:/Project/PROJ Dataset/Dataset.csv')
for i in df.columns:
  df.loc[df[i] == "Data missing or out of range", i] = None
#df['driver_imd_decile'] = df['driver_imd_decile'].fillna(df['driver_imd_decile'].mode()[0])  
#df['age_of_vehicle'].fillna(6, inplace=True)
#df['model'] = df['model'].fillna(df['model'].mode()[0])
#df['engine_capacity_cc'].fillna(1840, inplace=True)
#df['lsoa_of_accident_location'].fillna(method='ffill', inplace=True)
for i in df:
  if df[i].dtype =='object':
    df[i] = df[i].fillna(df[i].mode()[0])
  elif df[i].dtype=='float64':
     df[i] = df[i].fillna(df[i].mean())
  elif df[i].dtype=='datetime64[ns]':
    df[i] = df[i].fillna(df[i].mode()[0])




with st.sidebar:
    
    selected=option_menu("Road Accident Analysis",
                         ['Home',
                          'Predict using Model',
                          'Data Analysis'],
                         icons=['house-door-fill','arrow-right-circle','clipboard-data'],
                         default_index=0)
    
 
#Home Page
if selected=='Home':
    #Page Title
    img=Image.open(r'C:/Users/dell/OneDrive/Pictures/Road Accident analysis pictures/road.jpg')
    st.title('Road Accident Analysis🚦')
    st.markdown("\n")
    st.image(img)
    st.markdown('***')
    st.markdown('- ### **Introduction**')   
    st.markdown('Road accidents continue to be a leading cause of death, disabilities and hospitalization in the country despite our commitment and efforts. India ranks first in the number of road accident deaths across the 199 countries and accounts for almost 11% of the accident related deaths in the World.A total number of 449,002 accidents took place in the country during the calendar year 2019leading to 151,113 deaths and 451,361 injuries. In percentage terms, the number of accidents decreased by 3.86 % in 2019over that of the previous year, while the accident related deaths decreased by 0.20 % and the persons injured decreased by 3.86. ')
    st.markdown('\n')
    st.markdown('- ### **Objective**')
    st.markdown('As the number of accidents are increasing day by day our aim to to detect some factors that are responsible for these accidents.We are using a predefined dataset which has some attributes and we are going to detect the factors from those attributes. ')
    st.markdown('***')
    

 
#Predicting model
if selected=='Predict using Model':
    st.title('Predicting using Ml Models')
    model=st.sidebar.radio(" Select a model 🤗 :" ,['Model 1(Logistic_Regression)','Model 2(Random_Forest_Classifier)','Model 3(AdaBoost Classifier)','Model 4(Bagging Classifier)','Model 5(LogisticRegressionCV)','Model 6(RandomizedSearchCV)'])
    if model == 'Model 1(Logistic_Regression)':
        col1,col2,col3=st.columns(3)
        col4,col5=st.columns([1,3])
        col6,col7,col8=st.columns(3)
        col9,col10,col11=st.columns(3)
        col12,col13=st.columns([1,3])
        with col1:
            a = df['make'].unique()
            a.sort()
            i1 = st.selectbox("Select Company :",a)
        with col2:
            dfn = df[(df['make'] == i1)]
            b = dfn['model'].unique()
            b.sort()
            i2 = st.selectbox("Select model :",b)
        with col3:
            dfn1 = dfn[dfn['model'] == i2]
            c = dfn1['Vehicle_Type'].unique()
            i3=st.selectbox("Select vehicle Type :" ,c)
        with col4:
            #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
            i4=st.slider("Select Age of Vehicle: ",0,50,6)
        with col5:
            i5=st.slider("Select speed limit: ",0,120,30)
        with col6:
            d=df['Weather_Conditions'].unique()
            i6=st.selectbox("Select Weather Condition: ",d)
        with col7:
            e=df['Road_Type'].unique()
            i7=st.selectbox("Select Road Type: ",e)
        with col8:
            f=df['Light_Conditions'].unique()
            i8=st.selectbox('Select the light conditions: ',f)
        with col9:
            g=df['Pedestrian_Crossing-Human_Control'].unique()
            i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
        with col10:
            h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
            i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
        with col11:
            i=df['Road_Surface_Conditions'].unique()
            i11=st.selectbox('Select Road_Surface_Conditions: ',i)
        with col12:
            j=df['Age_Band_of_Driver'].unique()
            i12=st.selectbox('Select Age_Band_of_Driver: ',j)
        with col13:
            i13=st.radio('Select Gender: ',('Male','Female'))
            
        le_make = LabelEncoder()
        le_model = LabelEncoder()
        le_veh = LabelEncoder()
        le_ageVeh = LabelEncoder()
        le_speed = LabelEncoder()    
        le_weather = LabelEncoder()
        le_road = LabelEncoder()
        le_light = LabelEncoder()
        le_human = LabelEncoder()
        le_police = LabelEncoder()
        le_surface = LabelEncoder()
        le_driverage = LabelEncoder()
        le_gender = LabelEncoder()
        pred_acc=''
        acc_l=[]
        df['make'] = le_make.fit_transform(df['make'])
        df['model'] = le_model.fit_transform(df['model'])
        df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
        df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
        df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
        df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
        df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
        df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
        df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
        df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
        df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
        df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
        df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
        
        scalar=StandardScaler()
        xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
        X=scalar.fit_transform(xi)
        input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                               'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                               'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
        input_data['model'] = le_model.fit_transform(input_data['model'])
        input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
        input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
        input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
        input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
        input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
        input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
        input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
        input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
        input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
        input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
        input_data=scalar.transform(input_data)

        
        if st.button('Accident Severity Result: '):
            #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
            #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
            
            acc_pred=road_accident_model[0].predict(input_data)
            
            st.write('The accuracy of the model is: 85.38%')
            if acc_pred[0]==0:
                if i5>40:
                    acc_l.append('Speed Limit')
                    if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                        pred_acc='Accident is fatal'
                        acc_l.append('Weather Conditions')
                    if i7=='Slip Road' :
                        pred_acc='Accident is fatal'
                        acc_l.append('Road Type')
                    if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                        pred_acc='Accident is fatal'
                        acc_l.append('Lightning Conditions')
                    if i11=='Wet or damp' or i11=='Frost or ice' :
                        pred_acc='Accident is fatal'
                        acc_l.append('Road Surface condition')
                    if i10>1.0:
                        pred_acc='Accident is fatal'
                else:
                    pred_acc='Accident is not fatal'
                    if i6=='Raining+high winds' or i6=='Fog or mist' :
                        acc_l.append('Speed Limit')
                        acc_l.append('Weather Conditions')
                    elif i7=='Slip Road' :
                        acc_l.append('Road Type')
                    elif i8=='Darkness-no lightning' :
                        acc_l.append('Lightning Conditions')
                    elif i11=='Wet or Dry' or i11=='Frost or ice' :
                            acc_l.append('Road Surface condition')
                    elif i4>9:
                        acc_l.append('Age of Vehicle')
                    else:
                        acc_l.append('Weather Condithions')
                        acc_l.append('Road Type')
                        
            else:
                pred_acc='Accident is fatal'
                acc_l.append('Road Surface')
                if i5>40:
                    acc_l.append('Speed Limit')
                    if i6=='Raining+high winds' or i6=='Fog or mist' :
                        
                        acc_l.append('Weather Conditions')
                    elif i7=='Slip Road' :
                       acc_l.append('Road Type')
                    elif i8=='Darkness-no lightning' :
                       acc_l.append('Lightning Conditions')
                    elif i11=='Wet or Dry' or i11=='Frost or ice' :
                        acc_l.append('Road Surface condition')
                elif i4>9:
                    acc_l.append('Age of Vehicle')
                else:
                    acc_l.append('Weather Condithions')
                    acc_l.append('Road Type')    
            st.write('The possible factors for the accident could be:')
            for i in acc_l:
                st.write(i)
        st.success(pred_acc)
   
    if model == 'Model 2(Random_Forest_Classifier)':
         col1,col2,col3=st.columns(3)
         col4,col5=st.columns([1,3])
         col6,col7,col8=st.columns(3)
         col9,col10,col11=st.columns(3)
         col12,col13=st.columns([1,3])
         with col1:
             a = df['make'].unique()
             a.sort()
             i1 = st.selectbox("Select Company :",a)
         with col2:
             dfn = df[(df['make'] == i1)]
             b = dfn['model'].unique()
             b.sort()
             i2 = st.selectbox("Select model :",b)
         with col3:
             dfn1 = dfn[dfn['model'] == i2]
             c = dfn1['Vehicle_Type'].unique()
             i3=st.selectbox("Select vehicle Type :" ,c)
         with col4:
             #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
             i4=st.slider("Select Age of Vehicle: ",0,50,6)
         with col5:
             i5=st.slider("Select speed limit: ",0,120,30)
         with col6:
             d=df['Weather_Conditions'].unique()
             i6=st.selectbox("Select Weather Condition: ",d)
         with col7:
             e=df['Road_Type'].unique()
             i7=st.selectbox("Select Road Type: ",e)
         with col8:
             f=df['Light_Conditions'].unique()
             i8=st.selectbox('Select the light conditions: ',f)
         with col9:
             g=df['Pedestrian_Crossing-Human_Control'].unique()
             i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
         with col10:
             h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
             i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
         with col11:
             i=df['Road_Surface_Conditions'].unique()
             i11=st.selectbox('Select Road_Surface_Conditions: ',i)
         with col12:
             j=df['Age_Band_of_Driver'].unique()
             i12=st.selectbox('Select Age_Band_of_Driver: ',j)
         with col13:
             i13=st.radio('Select Gender: ',('Male','Female'))
             
         le_make = LabelEncoder()
         le_model = LabelEncoder()
         le_veh = LabelEncoder()
         le_ageVeh = LabelEncoder()
         le_speed = LabelEncoder()    
         le_weather = LabelEncoder()
         le_road = LabelEncoder()
         le_light = LabelEncoder()
         le_human = LabelEncoder()
         le_police = LabelEncoder()
         le_surface = LabelEncoder()
         le_driverage = LabelEncoder()
         le_gender = LabelEncoder()
         pred_acc1=''
         acc_l1=[]
         df['make'] = le_make.fit_transform(df['make'])
         df['model'] = le_model.fit_transform(df['model'])
         df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
         df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
         df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
         df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
         df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
         df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
         df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
         df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
         df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
         df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
         df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
         
         scalar=StandardScaler()
         xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
         X=scalar.fit_transform(xi)
         input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                                'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                                'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
         input_data['model'] = le_model.fit_transform(input_data['model'])
         input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
         input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
         input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
         input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
         input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
         input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
         input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
         input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
         input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
         input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
         input_data=scalar.transform(input_data)

         
         if st.button('Accident Severity Result: '):
             #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
             #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
             
             acc_pred1=road_accident_model[1].predict(input_data)
             
             st.write('The accuracy of the model is: 83.71%')
             if acc_pred1[0]==0:
                 if i5>40:
                     acc_l1.append('Speed Limit')
                     if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                         pred_acc1='Accident is fatal'
                         
                         acc_l1.append('Weather Conditions')
                     if i7=='Slip Road' :
                         pred_acc1='Accident is fatal'
                         acc_l1.append('Road Type')
                     if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                         pred_acc1='Accident is fatal'
                         acc_l1.append('Lightning Conditions')
                     if i11=='Wet or damp' or i11=='Frost or ice' :
                         pred_acc1='Accident is fatal'
                         acc_l1.append('Road Surface condition')
                     if i10>1.0:
                         pred_acc1='Accident is fatal'
                 else:
                     pred_acc1='Accident is not fatal'
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                        
                         acc_l1.append('Weather Conditions')
                     elif i7=='Slip Road' :
                         acc_l1.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                         acc_l1.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                             acc_l1.append('Road Surface condition')
                     elif i4>9:
                         acc_l1.append('Age of Vehicle')
                     else:
                         acc_l1.append('Weather Condithions')
                         acc_l1.append('Road Type')
                         
             else:
                 pred_acc1='Accident is fatal'
                 acc_l1.append('Road Surface')
                 if i5>40:
                     acc_l1.append('Speed Limit')
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                         
                         acc_l1.append('Weather Conditions')
                     elif i7=='Slip Road' :
                        acc_l1.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                        acc_l1.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                         acc_l1.append('Road Surface condition')
                 elif i4>9:
                     acc_l1.append('Age of Vehicle')
                 else:
                     acc_l1.append('Weather Condithions')
                     acc_l1.append('Road Type')    
             st.write('The possible factors for the accident could be:')
             for i in acc_l1:
                 st.write(i)
         st.success(pred_acc1)
         
    if model == 'Model 3(AdaBoost Classifier)':
        col1,col2,col3=st.columns(3)
        col4,col5=st.columns([1,3])
        col6,col7,col8=st.columns(3)
        col9,col10,col11=st.columns(3)
        col12,col13=st.columns([1,3])
        with col1:
            a = df['make'].unique()
            a.sort()
            i1 = st.selectbox("Select Company :",a)
        with col2:
            dfn = df[(df['make'] == i1)]
            b = dfn['model'].unique()
            b.sort()
            i2 = st.selectbox("Select model :",b)
        with col3:
            dfn1 = dfn[dfn['model'] == i2]
            c = dfn1['Vehicle_Type'].unique()
            i3=st.selectbox("Select vehicle Type :" ,c)
        with col4:
            #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
            i4=st.slider("Select Age of Vehicle: ",0,50,6)
        with col5:
            i5=st.slider("Select speed limit: ",0,120,30)
        with col6:
            d=df['Weather_Conditions'].unique()
            i6=st.selectbox("Select Weather Condition: ",d)
        with col7:
            e=df['Road_Type'].unique()
            i7=st.selectbox("Select Road Type: ",e)
        with col8:
            f=df['Light_Conditions'].unique()
            i8=st.selectbox('Select the light conditions: ',f)
        with col9:
            g=df['Pedestrian_Crossing-Human_Control'].unique()
            i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
        with col10:
            h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
            i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
        with col11:
            i=df['Road_Surface_Conditions'].unique()
            i11=st.selectbox('Select Road_Surface_Conditions: ',i)
        with col12:
            j=df['Age_Band_of_Driver'].unique()
            i12=st.selectbox('Select Age_Band_of_Driver: ',j)
        with col13:
            i13=st.radio('Select Gender: ',('Male','Female'))
            
        le_make = LabelEncoder()
        le_model = LabelEncoder()
        le_veh = LabelEncoder()
        le_ageVeh = LabelEncoder()
        le_speed = LabelEncoder()    
        le_weather = LabelEncoder()
        le_road = LabelEncoder()
        le_light = LabelEncoder()
        le_human = LabelEncoder()
        le_police = LabelEncoder()
        le_surface = LabelEncoder()
        le_driverage = LabelEncoder()
        le_gender = LabelEncoder()
        pred_acc2=''
        acc_l2=[]
        df['make'] = le_make.fit_transform(df['make'])
        df['model'] = le_model.fit_transform(df['model'])
        df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
        df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
        df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
        df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
        df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
        df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
        df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
        df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
        df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
        df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
        df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
        
        scalar=StandardScaler()
        xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
        X=scalar.fit_transform(xi)
        input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                               'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                               'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
        input_data['model'] = le_model.fit_transform(input_data['model'])
        input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
        input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
        input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
        input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
        input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
        input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
        input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
        input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
        input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
        input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
        input_data=scalar.transform(input_data)

        
        if st.button('Accident Severity Result: '):
            #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
            #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
            
            acc_pred2=road_accident_model[2].predict(input_data)
            
            st.write('The accuracy of the model is: 85.36%')
            if acc_pred2[0]==0:
                if i5>40:
                    acc_l2.append('Speed Limit')
                    if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                        pred_acc2='Accident is fatal'
                        
                        acc_l2.append('Weather Conditions')
                    if i7=='Slip Road' :
                        pred_acc2='Accident is fatal'
                        acc_l2.append('Road Type')
                    if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                        pred_acc2='Accident is fatal'
                        acc_l2.append('Lightning Conditions')
                    if i11=='Wet or damp' or i11=='Frost or ice' :
                        pred_acc2='Accident is fatal'
                        acc_l2.append('Road Surface condition')
                    if i10>1.0:
                        pred_acc2='Accident is fatal'
                else:
                    pred_acc2='Accident is not fatal'
                    if i6=='Raining+high winds' or i6=='Fog or mist' :
                        
                        acc_l2.append('Weather Conditions')
                    elif i7=='Slip Road' :
                        acc_l2.append('Road Type')
                    elif i8=='Darkness-no lightning' :
                        acc_l2.append('Lightning Conditions')
                    elif i11=='Wet or Dry' or i11=='Frost or ice' :
                            acc_l2.append('Road Surface condition')
                    elif i4>9:
                        acc_l2.append('Age of Vehicle')
                    else:
                        acc_l2.append('Weather Condithions')
                        acc_l2.append('Road Type')
                        
            else:
                pred_acc2='Accident is fatal'
                acc_l2.append('Road Surface')
                if i5>40:
                    acc_l2.append('Speed Limit')
                    if i6=='Raining+high winds' or i6=='Fog or mist' :
                        
                        acc_l2.append('Weather Conditions')
                    elif i7=='Slip Road' :
                       acc_l2.append('Road Type')
                    elif i8=='Darkness-no lightning' :
                       acc_l.append('Lightning Conditions')
                    elif i11=='Wet or Dry' or i11=='Frost or ice' :
                        acc_l2.append('Road Surface condition')
                elif i4>9:
                    acc_l2.append('Age of Vehicle')
                else:
                    acc_l2.append('Weather Condithions')
                    acc_l2.append('Road Type')    
            st.write('The possible factors for the accident could be:')
            for i in acc_l2:
                st.write(i)
        st.success(pred_acc2)
        
    if model == 'Model 4(Bagging Classifier)':
       col1,col2,col3=st.columns(3)
       col4,col5=st.columns([1,3])
       col6,col7,col8=st.columns(3)
       col9,col10,col11=st.columns(3)
       col12,col13=st.columns([1,3])
       with col1:
           a = df['make'].unique()
           a.sort()
           i1 = st.selectbox("Select Company :",a)
       with col2:
           dfn = df[(df['make'] == i1)]
           b = dfn['model'].unique()
           b.sort()
           i2 = st.selectbox("Select model :",b)
       with col3:
           dfn1 = dfn[dfn['model'] == i2]
           c = dfn1['Vehicle_Type'].unique()
           i3=st.selectbox("Select vehicle Type :" ,c)
       with col4:
           #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
           i4=st.slider("Select Age of Vehicle: ",0,50,6)
       with col5:
           i5=st.slider("Select speed limit: ",0,120,30)
       with col6:
           d=df['Weather_Conditions'].unique()
           i6=st.selectbox("Select Weather Condition: ",d)
       with col7:
           e=df['Road_Type'].unique()
           i7=st.selectbox("Select Road Type: ",e)
       with col8:
           f=df['Light_Conditions'].unique()
           i8=st.selectbox('Select the light conditions: ',f)
       with col9:
           g=df['Pedestrian_Crossing-Human_Control'].unique()
           i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
       with col10:
           h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
           i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
       with col11:
           i=df['Road_Surface_Conditions'].unique()
           i11=st.selectbox('Select Road_Surface_Conditions: ',i)
       with col12:
           j=df['Age_Band_of_Driver'].unique()
           i12=st.selectbox('Select Age_Band_of_Driver: ',j)
       with col13:
           i13=st.radio('Select Gender: ',('Male','Female'))
           
       le_make = LabelEncoder()
       le_model = LabelEncoder()
       le_veh = LabelEncoder()
       le_ageVeh = LabelEncoder()
       le_speed = LabelEncoder()    
       le_weather = LabelEncoder()
       le_road = LabelEncoder()
       le_light = LabelEncoder()
       le_human = LabelEncoder()
       le_police = LabelEncoder()
       le_surface = LabelEncoder()
       le_driverage = LabelEncoder()
       le_gender = LabelEncoder()
       pred_acc3=''
       acc_l3=[]
       df['make'] = le_make.fit_transform(df['make'])
       df['model'] = le_model.fit_transform(df['model'])
       df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
       df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
       df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
       df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
       df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
       df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
       df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
       df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
       df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
       df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
       df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
       
       scalar=StandardScaler()
       xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
       X=scalar.fit_transform(xi)
       input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                              'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                              'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
       input_data['model'] = le_model.fit_transform(input_data['model'])
       input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
       input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
       input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
       input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
       input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
       input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
       input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
       input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
       input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
       input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
       input_data=scalar.transform(input_data)

       
       if st.button('Accident Severity Result: '):
           #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
           #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
           
           acc_pred3=road_accident_model[3].predict(input_data)
           
           st.write('The accuracy of the model is: 85.38%')
           if acc_pred3[0]==0:
               if i5>40:
                   acc_l.append('Speed Limit')
                   if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                       pred_acc3='Accident is fatal'
                       acc_l3.append('Speed Limit')
                       acc_l3.append('Weather Conditions')
                   if i7=='Slip Road' :
                       pred_acc3='Accident is fatal'
                       acc_l3.append('Road Type')
                   if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                       pred_acc3='Accident is fatal'
                       acc_l3.append('Lightning Conditions')
                   if i11=='Wet or damp' or i11=='Frost or ice' :
                       pred_acc3='Accident is fatal'
                       acc_l3.append('Road Surface condition')
                   if i10>1.0:
                       pred_acc3='Accident is fatal'
               else:
                   pred_acc3='Accident is not fatal'
                   if i6=='Raining+high winds' or i6=='Fog or mist' :
                       acc_l3.append('Weather Conditions')
                   elif i7=='Slip Road' :
                       acc_l3.append('Road Type')
                   elif i8=='Darkness-no lightning' :
                       acc_l3.append('Lightning Conditions')
                   elif i11=='Wet or Dry' or i11=='Frost or ice' :
                           acc_l3.append('Road Surface condition')
                   elif i4>9:
                       acc_l3.append('Age of Vehicle')
                   else:
                       acc_l3.append('Weather Condithions')
                       acc_l3.append('Road Type')
                       
           else:
               pred_acc3='Accident is fatal'
               acc_l3.append('Road Surface')
               if i5>40:
                   acc_l3.append('Speed Limit')
                   if i6=='Raining+high winds' or i6=='Fog or mist' :
                       
                       acc_l3.append('Weather Conditions')
                   elif i7=='Slip Road' :
                      acc_l3.append('Road Type')
                   elif i8=='Darkness-no lightning' :
                      acc_l3.append('Lightning Conditions')
                   elif i11=='Wet or Dry' or i11=='Frost or ice' :
                       acc_l3.append('Road Surface condition')
               elif i4>9:
                   acc_l3.append('Age of Vehicle')
               else:
                   acc_l3.append('Weather Condithions')
                   acc_l3.append('Road Type')    
           st.write('The possible factors for the accident could be:')
           for i in acc_l3:
               st.write(i)
       st.success(pred_acc3)
        
    if model == 'Model 5(LogisticRegressionCV)':
         col1,col2,col3=st.columns(3)
         col4,col5=st.columns([1,3])
         col6,col7,col8=st.columns(3)
         col9,col10,col11=st.columns(3)
         col12,col13=st.columns([1,3])
         with col1:
             a = df['make'].unique()
             a.sort()
             i1 = st.selectbox("Select Company :",a)
         with col2:
             dfn = df[(df['make'] == i1)]
             b = dfn['model'].unique()
             b.sort()
             i2 = st.selectbox("Select model :",b)
         with col3:
             dfn1 = dfn[dfn['model'] == i2]
             c = dfn1['Vehicle_Type'].unique()
             i3=st.selectbox("Select vehicle Type :" ,c)
         with col4:
             #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
             i4=st.slider("Select Age of Vehicle: ",0,50,6)
         with col5:
             i5=st.slider("Select speed limit: ",0,120,30)
         with col6:
             d=df['Weather_Conditions'].unique()
             i6=st.selectbox("Select Weather Condition: ",d)
         with col7:
             e=df['Road_Type'].unique()
             i7=st.selectbox("Select Road Type: ",e)
         with col8:
             f=df['Light_Conditions'].unique()
             i8=st.selectbox('Select the light conditions: ',f)
         with col9:
             g=df['Pedestrian_Crossing-Human_Control'].unique()
             i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
         with col10:
             h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
             i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
         with col11:
             i=df['Road_Surface_Conditions'].unique()
             i11=st.selectbox('Select Road_Surface_Conditions: ',i)
         with col12:
             j=df['Age_Band_of_Driver'].unique()
             i12=st.selectbox('Select Age_Band_of_Driver: ',j)
         with col13:
             i13=st.radio('Select Gender: ',('Male','Female'))
             
         le_make = LabelEncoder()
         le_model = LabelEncoder()
         le_veh = LabelEncoder()
         le_ageVeh = LabelEncoder()
         le_speed = LabelEncoder()    
         le_weather = LabelEncoder()
         le_road = LabelEncoder()
         le_light = LabelEncoder()
         le_human = LabelEncoder()
         le_police = LabelEncoder()
         le_surface = LabelEncoder()
         le_driverage = LabelEncoder()
         le_gender = LabelEncoder()
         pred_acc4=''
         acc_l4=[]
         df['make'] = le_make.fit_transform(df['make'])
         df['model'] = le_model.fit_transform(df['model'])
         df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
         df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
         df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
         df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
         df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
         df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
         df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
         df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
         df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
         df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
         df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
         
         scalar=StandardScaler()
         xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
         X=scalar.fit_transform(xi)
         input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                                'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                                'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
         input_data['model'] = le_model.fit_transform(input_data['model'])
         input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
         input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
         input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
         input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
         input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
         input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
         input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
         input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
         input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
         input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
         input_data=scalar.transform(input_data)

         
         if st.button('Accident Severity Result: '):
             #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
             #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
             
             acc_pred4=road_accident_model[4].predict(input_data)
             
             st.write('The accuracy of the model is: 85.38%')
             if acc_pred4[0]==0:
                 if i5>40:
                     acc_l4.append('Speed Limit')
                     if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                         pred_acc4='Accident is fatal'
                         acc_l4.append('Weather Conditions')
                     if i7=='Slip Road' :
                         pred_acc4='Accident is fatal'
                         acc_l4.append('Road Type')
                     if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                         pred_acc4='Accident is fatal'
                         acc_l4.append('Lightning Conditions')
                     if i11=='Wet or damp' or i11=='Frost or ice' :
                         pred_acc4='Accident is fatal'
                         acc_l4.append('Road Surface condition')
                     if i10>1.0:
                         pred_acc4='Accident is fatal'
                 else:
                     pred_acc4='Accident is not fatal'
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                         acc_l4.append('Weather Conditions')
                     elif i7=='Slip Road' :
                         acc_l4.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                         acc_l4.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                             acc_l4.append('Road Surface condition')
                     elif i4>9:
                         acc_l4.append('Age of Vehicle')
                     else:
                         acc_l4.append('Weather Condithions')
                         acc_l4.append('Road Type')
                         
             else:
                 pred_acc4='Accident is fatal'
                 acc_l4.append('Road Surface')
                 if i5>40:
                     acc_l4.append('Speed Limit')
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                         
                         acc_l4.append('Weather Conditions')
                     elif i7=='Slip Road' :
                        acc_l4.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                        acc_l4.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                         acc_l4.append('Road Surface condition')
                 elif i4>9:
                     acc_l4.append('Age of Vehicle')
                 else:
                     acc_l4.append('Weather Condithions')
                     acc_l4.append('Road Type')    
             st.write('The possible factors for the accident could be:')
             for i in acc_l4:
                 st.write(i)
         st.success(pred_acc4)

    if model == 'Model 6(RandomizedSearchCV)':
         col1,col2,col3=st.columns(3)
         col4,col5=st.columns([1,3])
         col6,col7,col8=st.columns(3)
         col9,col10,col11=st.columns(3)
         col12,col13=st.columns([1,3])
         with col1:
             a = df['make'].unique()
             a.sort()
             i1 = st.selectbox("Select Company :",a)
         with col2:
             dfn = df[(df['make'] == i1)]
             b = dfn['model'].unique()
             b.sort()
             i2 = st.selectbox("Select model :",b)
         with col3:
             dfn1 = dfn[dfn['model'] == i2]
             c = dfn1['Vehicle_Type'].unique()
             i3=st.selectbox("Select vehicle Type :" ,c)
         with col4:
             #dfn2 = dfn1[dfn1['Age_of_Vehicle']==i3]
             i4=st.slider("Select Age of Vehicle: ",0,50,6)
         with col5:
             i5=st.slider("Select speed limit: ",0,120,30)
         with col6:
             d=df['Weather_Conditions'].unique()
             i6=st.selectbox("Select Weather Condition: ",d)
         with col7:
             e=df['Road_Type'].unique()
             i7=st.selectbox("Select Road Type: ",e)
         with col8:
             f=df['Light_Conditions'].unique()
             i8=st.selectbox('Select the light conditions: ',f)
         with col9:
             g=df['Pedestrian_Crossing-Human_Control'].unique()
             i9=st.selectbox('Select Pedestrian_Crossing-Human_Control: ',g)
         with col10:
             h=df['Did_Police_Officer_Attend_Scene_of_Accident'].unique()
             i10=st.selectbox('Select Did_Police_Officer_Attend_Scene_of_Accident: ',h)
         with col11:
             i=df['Road_Surface_Conditions'].unique()
             i11=st.selectbox('Select Road_Surface_Conditions: ',i)
         with col12:
             j=df['Age_Band_of_Driver'].unique()
             i12=st.selectbox('Select Age_Band_of_Driver: ',j)
         with col13:
             i13=st.radio('Select Gender: ',('Male','Female'))
             
         le_make = LabelEncoder()
         le_model = LabelEncoder()
         le_veh = LabelEncoder()
         le_ageVeh = LabelEncoder()
         le_speed = LabelEncoder()    
         le_weather = LabelEncoder()
         le_road = LabelEncoder()
         le_light = LabelEncoder()
         le_human = LabelEncoder()
         le_police = LabelEncoder()
         le_surface = LabelEncoder()
         le_driverage = LabelEncoder()
         le_gender = LabelEncoder()
         pred_acc5=''
         acc_l5=[]
         df['make'] = le_make.fit_transform(df['make'])
         df['model'] = le_model.fit_transform(df['model'])
         df['Vehicle_Type'] = le_veh.fit_transform(df['Vehicle_Type'])
         df['Age_of_Vehicle'] = le_ageVeh.fit_transform(df['Age_of_Vehicle'])
         df['Speed_limit'] = le_speed.fit_transform(df['Speed_limit'])
         df['Weather_Conditions'] = le_weather.fit_transform(df['Weather_Conditions'])
         df['Road_Type'] = le_road.fit_transform(df['Road_Type'])
         df['Light_Conditions'] = le_light.fit_transform(df['Light_Conditions'])
         df['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(df['Pedestrian_Crossing-Human_Control'])
         df['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
         df['Road_Surface_Conditions'] = le_surface.fit_transform(df['Road_Surface_Conditions'])
         df['Age_Band_of_Driver'] = le_driverage.fit_transform(df['Age_Band_of_Driver'])
         df['Sex_of_Driver'] = le_gender.fit_transform(df['Sex_of_Driver'])
         
         scalar=StandardScaler()
         xi=df[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
         X=scalar.fit_transform(xi)
         input_data = pd.DataFrame({'Did_Police_Officer_Attend_Scene_of_Accident': i10, 'Light_Conditions': i8, 'Pedestrian_Crossing-Human_Control': i9,
                                'Road_Surface_Conditions': i11,'Road_Type': i7, 'Speed_limit': i5, 'Weather_Conditions': i6,
                                'Age_Band_of_Driver': i12, 'Age_of_Vehicle':i4, 'Sex_of_Driver':i13,'model':i2 }, index=[1])
         input_data['model'] = le_model.fit_transform(input_data['model'])
         input_data['Age_of_Vehicle'] = le_ageVeh.fit_transform(input_data['Age_of_Vehicle'])
         input_data['Speed_limit'] = le_speed.fit_transform(input_data['Speed_limit'])
         input_data['Weather_Conditions'] = le_weather.fit_transform(input_data['Weather_Conditions'])
         input_data['Road_Type'] = le_road.fit_transform(input_data['Road_Type'])
         input_data['Light_Conditions'] = le_light.fit_transform(input_data['Light_Conditions'])
         input_data['Pedestrian_Crossing-Human_Control'] = le_human.fit_transform(input_data['Pedestrian_Crossing-Human_Control'])
         input_data['Did_Police_Officer_Attend_Scene_of_Accident'] = le_police.fit_transform(input_data['Did_Police_Officer_Attend_Scene_of_Accident'])
         input_data['Road_Surface_Conditions'] = le_surface.fit_transform(input_data['Road_Surface_Conditions'])
         input_data['Age_Band_of_Driver'] = le_driverage.fit_transform(input_data['Age_Band_of_Driver'])
         input_data['Sex_of_Driver'] = le_gender.fit_transform(input_data['Sex_of_Driver'])
         input_data=scalar.transform(input_data)

         
         if st.button('Accident Severity Result: '):
             #data = np.array([[i10,i8,i9,i11,i7,i5,i6,i12,i4,i13,i2]])
             #[['Did_Police_Officer_Attend_Scene_of_Accident','Light_Conditions','Pedestrian_Crossing-Human_Control','Road_Surface_Conditions','Road_Type','Speed_Limit','Weather_Conditions','Age_Band_of_Driver', 'Age_of_Vehicle','Sex_of_Driver','model']]
             
             acc_pred5=road_accident_model[5].predict(input_data)
             
             st.write('The accuracy of the model is: 85.38%')
             if acc_pred5[0]==0:
                 if i5>40:
                     acc_l5.append('Speed Limit')
                     if i6=='Raining + high winds' or i6=='Fog or mist' or i6=='Snowing + high winds':
                         pred_acc5='Accident is fatal'
                         acc_l5.append('Weather Conditions')
                     if i7=='Slip Road' :
                         pred_acc5='Accident is fatal'
                         acc_l5.append('Road Type')
                     if i8=='Darkness-no lightning' or i8=='Darkness-lights unlit' :
                         pred_acc5='Accident is fatal'
                         acc_l5.append('Lightning Conditions')
                     if i11=='Wet or damp' or i11=='Frost or ice' :
                         pred_acc5='Accident is fatal'
                         acc_l5.append('Road Surface condition')
                     if i10>1.0:
                         pred_acc5='Accident is fatal'
                 else:
                     pred_acc5='Accident is not fatal'
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                         acc_l5.append('Weather Conditions')
                     elif i7=='Slip Road' :
                         acc_l5.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                         acc_l5.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                             acc_l5.append('Road Surface condition')
                     elif i4>9:
                         acc_l5.append('Age of Vehicle')
                     else:
                         acc_l5.append('Weather Condithions')
                         acc_l5.append('Road Type')
                         
             else:
                 pred_acc5='Accident is fatal'
                 acc_l5.append('Road Surface')
                 if i5>40:
                     acc_l5.append('Speed Limit')
                     if i6=='Raining+high winds' or i6=='Fog or mist' :
                         acc_l5.append('Weather Conditions')
                     elif i7=='Slip Road' :
                        acc_l5.append('Road Type')
                     elif i8=='Darkness-no lightning' :
                        acc_l5.append('Lightning Conditions')
                     elif i11=='Wet or Dry' or i11=='Frost or ice' :
                         acc_l5.append('Road Surface condition')
                 elif i4>9:
                     acc_l5.append('Age of Vehicle')
                 else:
                     acc_l5.append('Weather Condithions')
                     acc_l5.append('Road Type')    
             st.write('The possible factors for the accident could be:')
             for i in acc_l5:
                 st.write(i)
         st.success(pred_acc5)
    
    
 #Data Analysis Part   
if selected=='Data Analysis':
    st.title('Analysis of Road Accidents')
    st.markdown("- Analyzing the data which has around `26k` data points. The goal is to get some insights about the data for model Building. ")    
    st.markdown("A small portion of data")
    df=df.drop_duplicates()
    st.dataframe(df.head(20))
    st.markdown('____')
    st.write(''' ### Freature Details:
             
              1.Accident_Index:Every accident has an unique index number
              2.1st Road Class:Represents the first road class
              3.1st Road number:Represents the first road number
              4.2nd Road Class:Represents the second road class
              5.2nd Road Number:Represents the second road number
              6.Accident severity:Represents how severe the accident was
              7.Carriageway Hazards:Represents whethe there was any carriageway damage or not
              8.Date:Represents on what date the accident took place
              9.Day of Week:Represents at what day the accident took place
              10.Did_Police_Officer_Attend_Scene_of_Accident:Represents whether the police attended 
                                                             the the accident area or not. 
              11.Junction control:Represents the junction control
              12.Junction Detail:Tell the detail of the junction
              13.Latitude:Tells the latitude of the accident
              14.Light Conditions:Represents the light conditions.
              15.Local_Authority_(District): Represents the local authority of the district.
              16.Local_Authority_(Highway):  Represents the local authority of the highway.
              17.Location_Easting_OSGR:Represnts the longitude 
              18.Location_Northing_OSGR:Represents the latitude
              19.LSOA_of_Accident_Location:Represents the LSOA of the accident Location.
              20.Longitute:Represents the longitude of the accident spot
              21.Number of Casualities:The total casualities involved in the accident.
              22.Number of Vehicles:Tells the number of vehicle involved in the accident
              23.Pedestrian_Crossing-Human_Control:Tells whether any pedetrian crossing the road while accident.
              24.Pedestrian_Crossing-Physical_Facilities:Tells whether any pedestrian crossed physical facilities.
              25.Police_Force:Tells of what area police was present.
              26.Road_Surface_Conditions:Tels about the condition of the road surface.
              27.Road Type:Tells about the type of the road.
              28.Special_Conditions_at_Site:Tells whether any special condition was present at the site.
              29.Time:At what time accident has occured.
              30.Urban or Rural Area:What was the area of the accident.
              31.Speed Limit:Tells the speed limit of the vehicle.
              32.Weather_Conditions:Tells about the weather condition.
              33.Age_Band_of_Driver:Tells about the age range of the driver.
              34.Year:Represnts at which year the accident ha taken place.
              35.Age band of vehicle:Tells how much old is the vehicle.
              36.InScotland:Whether the accident happend in scotland.
              37.Driver_Home_Area_Type:Tells about the home area of the driver.
              38.Driver_IMD_Decile: Tells about the driver imd decile.
              39.Hit_Object_in_Carriageway:Whether any object was hit in carriageway.
              40.Hit_Object_off_Carriageway:Whether any object was hit off carriageway.
              41.Journey_Purpose_of_Driver:Describes about the journey purpose of the driver.
              42.Junction_Location:Tells about the junction location
              43.Make:Tells about the makers of teh vehicle.
              44.Model:Tells about the model of the vehicle.
              45.Propulsion Code:Describes what fuel does the vehicle uses.
              46.Sex_of_Driver:Tells about the sex of the driver
              47.Skidding_and_Overturning:Describes whether the vehicle had skid or overturned.
              48.Towing_and_Articulation:Describes whether the vehicle had towed or articulated. 
              49.Vehicle_Leaving_Carriageway:Whether vehicle left any carriageway or not.
              50.Vehicle_Location.Restricted_Lane:Tells the vehicles loction in restricted lane.
              51.Vehicle_Manoeuvre:Whether there was any vehicle manoeuvre.
              52.Vehicle_Reference:Tells about the vehicle reference.
              53.Vehicle_Type:Describes the type of vehicle
              54.Was_Vehicle_Left_Hand_Drive:Was the driver lefty.
              55.1st_Point_of_Impact: Tells the first point of impact.
              ''')
              
              
    st.markdown('___')  

        
    st.write(''' #### Basic Information about data
             ##   Column                                       Non-Null Count  Dtype  
            ---  ------                                       --------------  -----  
            0   Accident_Index                               26406 non-null  object 
            1   1st_Road_Class                               26406 non-null  object 
            2   1st_Road_Number                              26406 non-null  float64
            3   2nd_Road_Class                               15971 non-null  object 
            4   2nd_Road_Number                              26196 non-null  float64
            5   Accident_Severity                            26406 non-null  object 
            6   Carriageway_Hazards                          26406 non-null  object 
            7   Date                                         26406 non-null  object 
            8   Day_of_Week                                  26406 non-null  object 
            9   Did_Police_Officer_Attend_Scene_of_Accident  26406 non-null  float64
            10  Junction_Control                             26406 non-null  object 
            11  Junction_Detail                              26406 non-null  object 
            12  Latitude                                     26406 non-null  float64
            13  Light_Conditions                             26406 non-null  object 
            14  Local_Authority_(District)                   26406 non-null  object 
            15  Local_Authority_(Highway)                    26406 non-null  object 
            16  Location_Easting_OSGR                        26406 non-null  float64
            17  Location_Northing_OSGR                       26406 non-null  float64
            18  Longitude                                    26406 non-null  float64
            19  LSOA_of_Accident_Location                    24042 non-null  object 
            20  Number_of_Casualties                         26406 non-null  float64
            21  Number_of_Vehicles                           26406 non-null  float64
            22  Pedestrian_Crossing-Human_Control            26406 non-null  float64
            23  Pedestrian_Crossing-Physical_Facilities      26406 non-null  float64
            24  Police_Force                                 26406 non-null  object 
            25  Road_Surface_Conditions                      26406 non-null  object 
            26  Road_Type                                    26406 non-null  object 
            27  Special_Conditions_at_Site                   26406 non-null  object 
            28  Speed_limit                                  26406 non-null  float64
            29  Time                                         26403 non-null  object 
            30  Urban_or_Rural_Area                          26406 non-null  object 
            31  Weather_Conditions                           26406 non-null  object 
            32  Year                                         26406 non-null  float64
            33  InScotland                                   26406 non-null  object 
            34  Age_Band_of_Driver                           15920 non-null  object 
            35  Age_of_Vehicle                               13969 non-null  float64
            36  Driver_Home_Area_Type                        15920 non-null  object 
            37  Driver_IMD_Decile                            7626 non-null   float64
            38  Engine_Capacity_.CC.                         14565 non-null  float64
            39  Hit_Object_in_Carriageway                    15920 non-null  object 
            40  Hit_Object_off_Carriageway                   15920 non-null  object 
            41  Journey_Purpose_of_Driver                    15920 non-null  object 
            42  Junction_Location                            15920 non-null  object 
            43  make                                         15920 non-null  object 
            44  model                                        15920 non-null  object 
            45  Propulsion_Code                              14883 non-null  object 
            46  Sex_of_Driver                                15920 non-null  object 
            47  Skidding_and_Overturning                     15920 non-null  object 
            48  Towing_and_Articulation                      15920 non-null  object 
            49  Vehicle_Leaving_Carriageway                  15920 non-null  object 
            50  Vehicle_Location.Restricted_Lane             15920 non-null  float64
            51  Vehicle_Manoeuvre                            15920 non-null  object 
            52  Vehicle_Reference                            15920 non-null  float64
            53  Vehicle_Type                                 15920 non-null  object 
            54  Was_Vehicle_Left_Hand_Drive                  15920 non-null  object 
            55  X1st_Point_of_Impact                         15920 non-null  object 
            56  Year.1                                       15920 non-null  float64''')
            
            
            
    st.header('Histogram for the number of accidents happening')      
    columns=['Date','Day_of_Week','Did_Police_Officer_Attend_Scene_of_Accident','Number_of_Vehicles',
         'Light_Conditions','Road_Surface_Conditions','Speed_limit','Urban_or_Rural_Area','Weather_Conditions',
         'Age_Band_of_Driver','Age_of_Vehicle','make','model','Propulsion_Code','Sex_of_Driver','Vehicle_Type']
    selected_feature=st.selectbox('Choose the feature',columns)
    for column in columns:
        if selected_feature == column:
            fig = px.histogram(df, x=column,color_discrete_sequence=['#8B1A1A'])
            fig.update_layout(
                      autosize=True,
                      width=1000,
                      height=500,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
            st.plotly_chart(fig)  
            
            
            
    st.header('Scatter plot with respect to number of accidents')
    columns_scatter=['Date','Day_of_Week','Did_Police_Officer_Attend_Scene_of_Accident','Number_of_Vehicles',
          'Light_Conditions','Road_Surface_Conditions','Speed_limit','Urban_or_Rural_Area','Weather_Conditions',
          'Age_Band_of_Driver','Age_of_Vehicle','make','model','Propulsion_Code','Sex_of_Driver','Vehicle_Type']   
    selected_scatterplot_feature=st.selectbox('Choose the Feature:',columns_scatter)
    for c in columns_scatter:
        if selected_scatterplot_feature== c:
            fig = px.scatter(data_frame=df,x=c,color_discrete_sequence=['#8B1A1A'])
            fig.update_layout(
                      autosize=True,
                      width=1000,
                      height=500,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
            st.plotly_chart(fig)
            
            
    st.header('Box plot for the number of accidents happening')
    columns_boxplot=['Day_of_Week','Did_Police_Officer_Attend_Scene_of_Accident','Number_of_Vehicles',
          'Light_Conditions','Road_Surface_Conditions','Speed_limit','Urban_or_Rural_Area','Weather_Conditions',
          'Age_Band_of_Driver','make','model','Propulsion_Code','Sex_of_Driver'] 
    selected_boxplot=st.selectbox('Choose the Feature:',columns_boxplot)  
    for co in columns_boxplot:
        if selected_boxplot == co:
          fig = px.box(df, x=co,color_discrete_sequence=['#8B1A1A'])
          fig.update_layout(
                      autosize=True,
                      width=1000,
                      height=500,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
          st.plotly_chart(fig)
          
              
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df['Date'] = le.fit_transform(df['Date'])
    df['Day_of_Week'] = le.fit_transform(df['Day_of_Week'])
    df['Did_Police_Officer_Attend_Scene_of_Accident'] = le.fit_transform(df['Did_Police_Officer_Attend_Scene_of_Accident'])
    df['Number_of_Vehicles'] = le.fit_transform(df['Number_of_Vehicles'])
    df['Light_Conditions'] = le.fit_transform(df['Light_Conditions'])
    df['Road_Surface_Conditions'] = le.fit_transform(df['Road_Surface_Conditions'])
    df['Speed_limit'] = le.fit_transform(df['Speed_limit'])
    df['Urban_or_Rural_Area'] = le.fit_transform(df['Urban_or_Rural_Area'])
    df['Weather_Conditions'] = le.fit_transform(df['Weather_Conditions'])
    df['Age_Band_of_Driver'] = le.fit_transform(df['Age_Band_of_Driver'])
    df['Age_of_Vehicle'] = le.fit_transform(df['Age_of_Vehicle'])
    df['make'] = le.fit_transform(df['make'])
    df['model'] = le.fit_transform(df['model'])
    df['Propulsion_Code'] = le.fit_transform(df['Propulsion_Code'])
    df['Sex_of_Driver'] = le.fit_transform(df['Sex_of_Driver'])
    df['Vehicle_Type'] = le.fit_transform(df['Vehicle_Type'])
    
    
    
    st.header("Heatmap for knowing the correlation.")
    fig, ax = plt.subplots( figsize=(15, 15))
    sns.set_context('paper',font_scale=1)
    sns.heatmap(df.corr(), ax=ax,annot=True)
    st.write(fig)
    
    
    
