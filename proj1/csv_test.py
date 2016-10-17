import csv
import numpy as np

with open('training_modified.csv', 'r') as raw_file:
    reader = csv.reader(raw_file, delimiter=',')
    raw_data = np.array(list(reader)[1:])

print(raw_data)
print(raw_data.shape)

"""
with open('result.csv', 'w') as result_file:
    writer = csv.writer(result_file)
    writer.writerows(tmp)
"""
