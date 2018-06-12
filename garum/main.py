import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold


def main():
   taxdf = pd.read_csv('../resources/resultsTax.txt', sep='\t', names=range(60))
   neighdf = pd.read_csv('../resources/resultsNeig.txt', sep='\t', names=range(60))
   specdf = pd.read_csv('../resources/resultsSpec.txt', sep='\t', names=range(60))

   groundTruth = pd.read_csv('/home/nacho/PycharmProjects/GARUM/resources/cessm2008_benchmark.txt', sep='\t')
   groundTruth = groundTruth.loc[:, ['protein1', 'protein2', 'SeqSim']]

   nbins = 10
   taxdf_hist = pd.DataFrame(np.apply_along_axis(lambda a: np.histogram(a, bins=nbins, density=True, range=(0,1))[0], 1,
                                    taxdf.iloc[:, 2:]))
   neighdf_hist = pd.DataFrame(np.apply_along_axis(lambda a: np.histogram(a, bins=nbins, density=True, range=(0, 1))[0], 1,
                                   neighdf.iloc[:, 2:]))
   specdf_hist = pd.DataFrame(np.apply_along_axis(lambda a: np.histogram(a, bins=nbins, density=True, range=(0, 1))[0], 1,
                                      specdf.iloc[:, 2:]))
   training_dataset = groundTruth
   training_dataset = pd.concat([training_dataset, taxdf_hist, neighdf_hist, specdf_hist], axis=1)

   regressor = SVR()

   kf = KFold(10)
   mean = 0
   for train, test in kf.split(training_dataset):
       regressor.fit(training_dataset.iloc[train, 3:], training_dataset.iloc[train, :]['SeqSim'])
       pred = regressor.predict(training_dataset.iloc[test, 3:])

       mean += np.corrcoef(pred, training_dataset.iloc[test, :]['SeqSim'])[0, 1]
   print(mean/10)


if "__main__" == __name__:
    main()