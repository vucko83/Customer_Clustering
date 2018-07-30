import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

'''
Read and prepare data
'''

def prepare(path_data, path_sifarnik , scaling = False):
    transactions = pd.read_csv(path_data)
    transactions= transactions.fillna(0)
    transactions = transactions.iloc[:, 1:]

    sifarnik = pd.read_csv(path_sifarnik)
    sifarnik['groupSKU'] = sifarnik['groupSKU'].map(lambda x: x.split(' -')[0])

    if scaling == True:

        scaler=MinMaxScaler(feature_range = (1,10)) # initialize scaler
        attributes = transactions.columns
        transactions = scaler.fit_transform(transactions)
        transactions = pd.DataFrame(transactions)
        transactions.columns = attributes

    transformer = TfidfTransformer().fit(transactions)

    test = transformer.transform(transactions)

    proizvodi = list(transactions.columns)

    return (test, transactions, proizvodi, sifarnik)


'''
pca = PCA(n_components=20)
test_pca = pca.fit_transform(test.todense())
'''
