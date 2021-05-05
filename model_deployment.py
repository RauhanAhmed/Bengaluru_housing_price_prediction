from pywebio.input import *
from pywebio.output import *
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split



df = pickle.load(open(r'C:\Users\xyz\Desktop\clean_dataset.pkl', 'rb'))

x_train, x_test, y_train, y_test = train_test_split(df.drop('price', axis = 1), df['price'], test_size = 0.1)

model = Lasso(alpha = 1)
model.fit(x_train, y_train)

def predict_price(location, total_sqft, bath, bhk):
    if location in df.columns:
        i = list(df.drop('price', axis = 1).columns).index(location)
        x = np.zeros(len(list(df.drop('price', axis = 1).columns)))
        x[0] = bhk
        x[1] = total_sqft
        x[2] = bath
        x[i] = 1
        return model.predict([x])[0]
    elif location in other_locations.index:
        i = list(df.drop('price', axis = 1).columns).index('other')
        x = np.zeros(len(list(df.drop('price', axis = 1).columns)))
        x[0] = bhk
        x[1] = total_sqft
        x[2] = bath
        x[i] = 1
        return model.predict([x])[0]
    else: 
        pass 


def pred():
    name = input('Please enter your name :',type = TEXT)
    location = select('Please chose your desired location from the drop-down.', df.columns[4:])
    area = input('Enter desired area in square ft.', type = FLOAT)
    bathrooms = input('Enter desired number of bathrooms', type = NUMBER)
    bhks = input("Enter the number of BHK's you want", type = NUMBER)
    put_markdown('## Hello {}'.format(name))
    put_processbar('bar')
    for i in range(1, 11):
        set_processbar('bar', i / 10)
        time.sleep(0.1)
    put_image(r'https://blog.hubspot.com/hubfs/Sales_Blog/real-estate-business-compressor.jpg')
    put_text('The location you have chosen is :', location\
         ,'\nThe area in square ft. you entered is :', area\
         ,'\n The number of bathrooms you entered are :',bathrooms\
         ,"\nThe number of BHK's chosen by you are :",bhks\
         ,'\n\n\n\nThe expected price for the house with the specifications you enetered according to our Machine Learning model is : Rs.', (predict_price(str(location),area,bathrooms,bhks)) * 100000)
    put_text('model accuracy is :', model.score(x_test, y_test))


if __name__ == '__main__':
    pred()