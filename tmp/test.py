import numpy as np
import random

authors = ["EAP","HPL","MWS"]

lstm = dict()
with open("kaggle_out.csv") as f:
    f.readline()
    for line in f:
        probs = list(map(eval, line.split(",")[1:]))
        id = line.split(",")[0]

        lstm[id] = authors[np.argmax(probs).item()]


logistic = dict()

with open("logistic_kaggle_out.csv") as f:
    f.readline()
    for line in f:
        probs = list(map(eval, line.split(",")[1:]))
        id = line.split(",")[0]
        logistic[id] = authors[np.argmax(probs).item()]


count = 0
for key in logistic.keys():
    if lstm[key] == logistic[key]:
        count += 1

print(count / len(logistic))

with open("../data/spooky-authorship/train.csv") as f:
    f.readline()
    lines = f.read().split("\n")
    random.shuffle(lines)

    select = lines[:int(len(lines) / 2)]

with open("../data/spooky-authorship/test.csv") as f:
    with open("new_data2.csv", "w") as g:
        g.write("id,text,author")

        for line in select:
            g.write(line.strip() + "\n")

        f.readline()
        for line in f:
            id = line.split(",")[0].replace("\"", "")
            if lstm[id] == logistic[id]:
                g.write(line.strip() + "," + lstm[id] + "\n")





