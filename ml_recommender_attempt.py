
import pandas as pd
import numpy as np
import torch
import random
import pickle
import sys

## LISA'S SELENIUM CODE


## DATA ARRANGING PART

all_orders_products = pd.read_csv('C:\\Users\\avikr\\Downloads\\instacart_data\\order_products__prior\\order_products__prior.csv', nrows = 200000)
print('read1')
all_products = pd.read_csv('C:\\Users\\avikr\\Downloads\\instacart_data\\products\\products.csv')
print('read2')
all_orders = pd.read_csv('C:\\Users\\avikr\\Downloads\\instacart_data\\orders\\orders.csv')
print('read3')

def get_product_info(user, product, price = None): # REPLACE
    if price == None: price = random.random()*8
    return [user, product, price]

def order_to_products():
    ret = {}
    loop_iter = 0
    for values in all_orders_products.iterrows():
        if loop_iter % 1000 == 0: print('loop 0', loop_iter)
        vs = values[1].values
        order, product = vs[0], vs[1]
        if order in ret.keys():
            ret[order].append(get_product_info(product))
        else:
            ret[order] = [get_product_info(product)]
        loop_iter += 1
    return ret

def dictionary_save(data, fname):
    with open(fname + '.pickle', 'wb') as f:
        # higher protocol == faster
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def dictionary_recover(option):
    if option == 1:
        return pickle.load(open('order_to_productinfos.pickle', 'rb'))
    
    TRAIN_DF_PICKLE = pickle.load(open('user_orders.pickle', 'rb'))
    return TRAIN_DF_PICKLE[0]

order_products = order_to_products() if sys.argv[1] == '1' else dictionary_recover(1)

if sys.argv[1] == '1':
    dictionary_save(order_products, 'order_to_productinfos')

TRAIN_DATAS = {}
TRAIN_DF_PICKLE = None

def random_choice_except (A, exclude):
    a = random.choice(A)
    while a == exclude:
        a = random.choice(A)
    return a

def verfiy_ordered_product(A, user, product):
    if user not in A.keys():
        return False
    located = False
    for order in A[user].keys():
        if product in A[user][order]:
            located = True
            break
    return located

def arrange_data():
    # user_id = 1
    # user -> orders -> product infos in order
    loop_iter = 0
    for values in all_orders.iterrows():
        if loop_iter % 5000 == 0: print('loop 1', loop_iter)
        vals = values[1].values
        order, user_id = vals[0], vals[1]
        order_data = order_products[order] if order in order_products.keys() else None # all the produts
        if order_data:
            if user_id in TRAIN_DATAS.keys():
                if order not in TRAIN_DATAS[user_id].keys():
                    TRAIN_DATAS[user_id][order] = order_data
            else:
                TRAIN_DATAS[user_id] = {}
                TRAIN_DATAS[user_id][order] = order_data
        loop_iter += 1

    loop_iter = 0
    all_products_and_fakes = []
    users = list(TRAIN_DATAS.keys())
    fake_users = [random_choice_except(user) for user in users]
    for user, fake_user in zip(users, fake_users):
        if loop_iter % 1000 == 0: print('loop 2', loop_iter)
        all_products_for_user = []
        user_products_added = 0
        for order in TRAIN_DATAS[user].keys():
            new_products = [product_info + [1] for product_info in TRAIN_DATAS[user][order]]
            all_products_for_user.extend(new_products)
        for order in TRAIN_DATAS[fake_user].keys():
            new_fake_products = [[user] + product_info[1:] + [0] for product_info in TRAIN_DATAS[fake_user][order]]
            all_products_for_user.extend(new_fake_products)
        all_products_and_fakes.extend(all_products_for_user)
        # TRAIN_DATAS[user] = pd.DataFrame(all_products_for_user, columns = ['user','product','price','label'])
        loop_iter += 1

    TRAIN_DF = pd.DataFrame(all_products_and_fakes, columns = ['user', 'product', 'price', 'label'])
    TRAIN_DF_PICKLE = {}
    TRAIN_DF_PICKLE[0] = TRAIN_DF

    print('finished arranging')

if sys.argv[2] == '1':
    arrange_data()
    dictionary_save(TRAIN_DF_PICKLE, 'user_orders')
    TRAIN_DATAS = TRAIN_DF_PICKLE[0]
else: 
    TRAIN_DATAS = dictionary_recover(2)


## ML Part

class ProductDataset(torch.utils.data.Dataset):
    def __init__(self, A):
        self.A = A.to_numpy()
        self.X = self.A[:,:-1] # all rows, all columns excluding the last
        self.Y = self.A[:,-1:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def to_dl(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size = 256, shuffle = True, num_workers = 4)

# TRAIN_DATASETS = [ProductDataset(user) for user in TRAIN_DATAS.keys()]

class NN(torch.nn.Module):
    def __init__(self):
        self.lin1, self.relu1 = torch.nn.Linear(5,10). torch.nn.ReLU()
        self.lin2, self.relu2 = torch.nn.Linear(10,20), torch.nn.ReLU()
        self.lin3, self.relu3 = torch.nn.Linear(20,10), torch.nn.ReLU()
        self.lin4 = torch.nn.Linear(10,2)
        self.network = torch.nn.Sequential(self.lin1, self.relu1, self.lin2, self.relu2,
                                                self.lin3, self.relu3, self.lin4)

    def forward(self, x):
        return self.network(x)

# use stochastic gradient descent and cross entropy loss
# network = NN()
# criterion, optimizer = torch.nn.CrossEntropyLoss(), torch.optim.SGD(network.parameters()) 

# model and losses for user dataset_id
def run_model(network, criterion, optimizer, dataset_id, epochs):

    traindata = to_dl(TRAIN_DATASETS[dataset_id])

    def run_one_epoch(traindata):
        epoch_loss = 0
        for batch_id, (data, y) in enumerate(traindata):
            optimizer.zero_grad()
            data.to(device)
            y.to(y)
            # get predictions, loss
            yhat = network(data.float())
            loss = criterion(yhat, y)
            epoch_loss += loss.item()
            # adjust params with gradients from backprop
            loss.backward()
            optimizer.step()

        epoch_loss /= len(traindata)
        return epoch_loss

    train_losses = [] 
    for epoch in range(epochs):
        epoch_loss = run_one_epoch(traindata)
        train_losses.append(epoch_loss)

    return network, train_losses



def bmr_calc(ht,wt,age,gender):
    bmr = None
    if gender == 'male':
        bmr = 66.47 + (13.75 * wt) + (5.003 * ht) - (6.755 * age)
    else:
        bmr = 655.1 + (9.563 * wt) + (1.85 * ht) - (4.676 * age)
    return bmr


