from math import prod
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt


def get_products():
    '''
    laods the data from the csv file and prints the first 5 entries
    '''
    products = pd.read_csv("Products.csv", lineterminator="\n",index_col= 0)
    print(products.head())
    return(products)

def format_datatypes(products):
    '''
    Changes the data types of columns to the appropriate one
    '''
    products['price'] = products['price'].str.replace('£','')
    products['price'] = products['price'].str.replace(',','')
    products['price'] = products['price'].astype('float64')
    products['page_id'] = products['page_id'].astype('int64')
    products['create_time'] = products['create_time'].astype('datetime64[ns]')
    # print(products.info())
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(products.head())
    return(products)

def get_msno_matrix(df):
    fig = msno.matrix(df)
    fig_copy = fig.get_figure()
    fig_copy.savefig('plot.png', bbox_inches = 'tight')

def remove_missing_values(products):
    '''
    prints the input dataframes info, removes the missing values or entries that should contain the alphabet but dont, and then prints the info again.
    '''
    print(products.info())
    products_copy = products[products['product_description'] != '']
    products_copy= products_copy[products_copy['product_description'].astype(str).str.contains('[A-Za-z]')]
    products_copy = products_copy[products_copy['location'] != '']
    products_copy = products_copy[products_copy['location'].astype(str).str.contains('[A-Za-z]')]
    products_copy = products_copy[products_copy['product_name'] != '']
    products_copy = products_copy[products_copy['product_name'].astype(str).str.contains('[A-Za-z]')]
    print(products_copy.info())

    return(products_copy)

if __name__ == '__main__':
    products = get_products()
    products = format_datatypes(products)
    products = remove_missing_values(products)
    products.to_csv('Products_formated.csv')
    
   
