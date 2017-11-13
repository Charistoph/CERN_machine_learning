import csv

with open('output.csv', 'r') as f:
    csvread = csv.reader(f)
    batch_data = list(csvread)

for item in enumerate(batch_data):
    if (item[1] == '-1'):
        print item[1]
