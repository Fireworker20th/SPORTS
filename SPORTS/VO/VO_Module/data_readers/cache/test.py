import pickle


with open("VKitti2.pickle","rb") as f:
    x = pickle.load(f)
    print(x)