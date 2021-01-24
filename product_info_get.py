
import pandas as pd
import numpy as np
import pickle

product_fname = 'C:\\Users\\avikr\\Downloads\\instacart_data\\products\\products.csv'
PRODUCTS_DF = pd.read_csv(product_fname)
PRODUCS_W_PRICES_DF = PRODUCTS_DF[['product_id','product_name']]

def colvect_to_vect(v):
	return v.reshape(v.shape[0])

# in: string for product query
# out: average top hit price
def selenium_script(product_query):
	# LISA's CODE HERE
	return prices # REPLACE

PRODUCTS_W_PRICES_DF['price'] = [selenium_script(product) for product in colvect_to_vect(PRODUCTS_DF[['product_name']])] 
PRODUCTS_W_PRICES_DF.to_csv(r'./products_prices.csv', index = False)


