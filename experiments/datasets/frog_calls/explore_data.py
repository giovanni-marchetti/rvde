import numpy as np
import csv
from torch.utils.data import Dataset

res = []
species = []
with open('Frogs.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for idx, row in enumerate(list(spamreader)[1:]):
        res.append([])
        for value in row[0].split(',')[1:-4]:
            res[idx].append(float(value))
        species.append(row[0].split(',')[-2])

#print(res, classes)
# print(len(res), len(species))
# print(len(set(species)))

species_dict = {}
for sp, i in zip(list(set(species)), range(len(set(species)))):
    species_dict[sp] = i

    print(species_dict)

species_num = [species_dict[key] for key in species]
# print(species_num)

# #SAVING ARRAYS
# res_numpy = np.array(res)
# lbls_numpy = np.array(species_num)
# print(lbls_numpy.shape, res_numpy.shape)
# np.save('frogs_data.npy', res_numpy)
# np.save('frogs_lbls.npy', lbls_numpy)
