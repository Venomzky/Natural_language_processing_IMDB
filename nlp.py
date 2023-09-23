import numpy as np

with open("reviews.txt") as f:
    raw_reviews = f.readlines()
with open("labels.txt") as f:
    raw_labels = f.readlines()

def sigmoid(x):
    return 1/(1+np.exp(-x))

tokens = list(map(lambda x:set(x.split(" ")),raw_reviews))
alpha, iteration = (0.01, 2)
hidden_size = 100



vocab = set()
for sentences in raw_reviews:
    words = sentences.split(' ')
    for word in words:
        if(len(word)>0):
            vocab.add(word)
vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word]=i

input_data = list()
for sentences in raw_reviews:
    sentences_to_index = list()
    for words in sentences.split(" "):
        try:
            sentences_to_index.append(word2index[words])
        except:
            ""
    input_data.append(sentences_to_index)

target_dataset = list()
for label in raw_labels:
    if label == "positive\n":
        target_dataset.append(1)
    else:
        target_dataset.append(0)


weights_0_1 = 0.2*np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2*np.random.random((hidden_size, 1)) - 0.1


correct, total = (0,0)
for iter in range(2):

    for i in range(len(input_data)-1000):
        x,y = (input_data[i], target_dataset[i])
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)
        weights_0_1[x] -= layer_1_delta*alpha
        weights_1_2 -=  np.outer(layer_1,layer_2_delta) *alpha

        if(abs(layer_2_delta)<0.5):
            correct += 1
        total +=1
        if(i%1000 == 0):
            print(f"Iter: {iter}")
            print(f"progress {i/float(len(input_data))}")
            print(f"acc {correct/total}")
        

correct,total = (0,0)
for i in range(len(input_data)-1000,len(input_data)):

       x = input_data[i]
       y = target_dataset[i]

       layer_1 = sigmoid(np.sum(weights_0_1[x],axis=0))
       layer_2 = sigmoid(np.dot(layer_1,weights_1_2))

       if(np.abs(layer_2 - y) < 0.5):
            correct += 1
       total += 1
print("-----------")
print("Test Accuracy:" + str(correct / float(total)))


from collections import Counter
import math

def similar(target):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - (weights_0_1[target_index])
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)
print("--------------")
print("Similarity:")
x = similar("terrible")
for i in x:
    print(i)











