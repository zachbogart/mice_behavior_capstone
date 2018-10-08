import pickle

file = open("results.pkl", "rb")
qwe = pickle.load(file)

print(qwe)