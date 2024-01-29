import pickle

def save_pickled_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickled_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data 
