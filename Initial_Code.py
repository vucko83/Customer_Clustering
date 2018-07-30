import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import Prepare_Data
from Prepare_Data import *
from sklearn.metrics import silhouette_score
import random

random.seed(1589)
import pickle

'''
Create Clusters: K-MEANS

'''






'''
best_silhouette = -np.inf
best_model = None
for n_clust in range(2,6):
    k_means = KMeans(n_clusters = n_clust, init='k-means++', max_iter= 500000, n_init= 3, n_jobs=4).fit(test)
    clusters = k_means.predict(test)
    if silhouette_score(test, clusters)>best_silhouette:
        best_model = k_means

clusters = best_model.predict(test)
transactions['Cluster'] = clusters
cluster_counts = transactions['Cluster'].value_counts()
cluster_counts

'''

path_parts = ['premium', 'gold', 'silver', 'standard']



for path_part in path_parts:

    test, transactions,  proizvodi, sifarnik = prepare(path_data = path_part+"_trans.csv", path_sifarnik='sifarnikGrupeNovo.csv', scaling= True)

    for n_clust in range(2,6):
        k_means = KMeans(n_clusters = n_clust, init='k-means++', max_iter= 500000, n_init= 3, n_jobs=4).fit(test)
        filename = 'Models/' + path_part + '_' +str(n_clust) +'.sav'
        pickle.dump(k_means, open(filename, 'wb'))







loaded_model = pickle.load(open('Models/premium_2.sav', 'rb'))

loaded_model.predict(test)

clusters = k_means.predict(test)
transactions['Cluster'] = clusters
cluster_counts = transactions['Cluster'].value_counts()
cluster_counts





'''
Visualize Cluster Centers
'''
centers = k_means.cluster_centers_[4]

df=pd.DataFrame({'x': proizvodi, 'y': centers })

df_sorted = df.sort_values('y', ascending=False)


df_sorted = df_sorted.set_index('x').join(sifarnik.set_index('groupID'))


centro_cloud = df_sorted.set_index('groupSKU')

tuples_dict = centro_cloud.to_dict()

tuples_dict_for_vis = tuples_dict['y']


wordcloud = WordCloud(relative_scaling=0.8, max_words = 200, max_font_size= 30, background_color='white').generate_from_frequencies(tuples_dict_for_vis)

#wordcloud = WordCloud(background_color='white').generate_from_frequencies(tuples_dict_for_vis)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('Figures/initial_fig')
plt.show()








'''
Create Clusters: Hierarchical

'''


from sklearn.cluster import AgglomerativeClustering

clt = AgglomerativeClustering(linkage='complete',
                              affinity='cityblock',
                              n_clusters=3)

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

