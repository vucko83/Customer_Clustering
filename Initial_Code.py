import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


from sklearn.mixture.gmm import GMM

'''
Read and prepare data
'''


transactions = pd.read_csv("premium_trans.csv")
transactions= transactions.fillna(0)
transactions = transactions.iloc[:, 1:]


sifarnik = pd.read_csv('sifarnikGrupaProizvoda.csv', encoding='cp1252')

sifarnik['groupSKU'] = sifarnik['groupSKU'].map(lambda x: x.split(' -')[0])


scaler=MinMaxScaler(feature_range = (1,10)) # initialize scaler


attributes = transactions.columns


transactions=scaler.fit_transform(transactions)

transactions =pd.DataFrame(transactions)


transactions.columns = attributes


transformer = TfidfTransformer().fit(transactions)


test = transformer.transform(transactions)


pca = PCA(n_components=20)
test_pca = pca.fit_transform(test.todense())


proizvodi = list(transactions.columns)

'''
Create Clusters: K-MEANS

'''

k_means = KMeans(n_clusters = 3, init='k-means++', max_iter= 10000, n_init= 3, n_jobs=4).fit(test)
clusters = k_means.predict(test)
transactions['Cluster']= clusters
cluster_counts = transactions['Cluster'].value_counts()
cluster_counts



#proizvodi.remove('Cluster')



'''
Visualize Cluster Centers
'''

centers = k_means.cluster_centers_[0]


df=pd.DataFrame({'x': proizvodi, 'y': centers })



plt.plot(x='x', y='y', data =df, color='skyblue' )
plt.show()


df_sorted = df.sort_values('y', ascending=False)





'''
Create Clusters: Hierarchical

'''


from sklearn.cluster import AgglomerativeClustering

clt = AgglomerativeClustering(linkage='ward',
                              affinity='euclidean',
                              n_clusters=2)

model = clt.fit(test.todense())


transactions.columns


transactions['Cluster'] = model.labels_

cluster_counts = transactions['Cluster'].value_counts()

cluster_counts



'''
Create Clusters - GMM
'''


np.set_printoptions(suppress=True)
from sklearn import mixture


test_dense = test.todense()
lowest_bic = np.infty # init bic to infinity (lower values are better)
bic = []
n_components_range = range(2, 10)
#cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = [ 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # select good initial means based on k-means++ method
        #kmeans_plus= KMeans(n_clusters=n_components, init='k-means++', random_state=0, n_init=10, max_iter=100).fit(test_dense)
        #init_means=kmeans_plus.cluster_centers_
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, max_iter=5000,  init_params='kmeans')
        gmm.fit(test_dense)
        # append bic performance and track best model
        bic.append(gmm.bic(test_dense))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm


best_gmm.n_components

predictions = best_gmm.predict_proba(test_dense)

clusters_gmm = predictions.argmax(axis=1)

transactions['Cluster'] = clusters_gmm

cluster_counts = transactions['Cluster'].value_counts()


cluster_counts

'''
Print WordCloud
'''

cluster_selection = transactions['Cluster'] == 1


# From counts


agg = test[cluster_selection]

# From TF_IDF not Counts

agg = pd.DataFrame(agg.todense())


agg.columns=proizvodi


#agg = agg.iloc[:, 1:-1].sum(axis=0)

agg = agg.median(axis=0)

new_agg=agg.reset_index()

new_agg.columns = ['groupID', 'Count']


values_for_filter = ['']
#values_for_filter = ['KAFA', 'VODA NEGAZIRANA', 'POLUTRAJNA ROBA', 'JOGURT NATUR', 'MLEKO SVEŽE', "MLEKO UHT 1L"]

agg_for_word_cloud = new_agg.set_index('groupID').join(sifarnik.set_index('groupID'))
agg_filtered = agg_for_word_cloud.loc[~agg_for_word_cloud['groupSKU'].isin(values_for_filter)]


agg_filtered = agg_filtered.set_index('groupSKU')

tuples_dict = agg_filtered.to_dict()

tuples_dict_for_vis = tuples_dict['Count']


wordcloud = WordCloud(relative_scaling=0.8, max_words = 200, max_font_size= 20, background_color='black').generate_from_frequencies(tuples_dict_for_vis)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

agg_filtered.sort_values(by=['Count'], ascending=False).head(100)


'''
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

cluster_selection = transactions['Cluster'] == 2

mb_data = transactions[cluster_selection].iloc[:,:-1]


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

mb_data = mb_data.applymap(encode_units)


frequent_itemsets = apriori(mb_data, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()


'''

