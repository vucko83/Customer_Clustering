import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.mixture.gmm import GMM

'''
Read and prepare data
'''


transactions = pd.read_csv("premium_trans.csv")
transactions= transactions.fillna(0)
transactions = transactions.iloc[:, 1:]


sifarnik = pd.read_csv('sifarnikGrupaProizvoda.csv', encoding='cp1252')

sifarnik['groupSKU'] = sifarnik['groupSKU'].map(lambda x: x.split(' -')[0])


transformer = TfidfTransformer().fit(transactions)


test = transformer.transform(transactions)



'''
Create Clusters: K-MEANS

'''

k_means = KMeans(n_clusters = 4, init='k-means++', max_iter= 10000, n_init= 10, n_jobs=4).fit(test)
clusters = k_means.predict(test)
transactions['Cluster']= clusters
cluster_counts = transactions['Cluster'].value_counts()
cluster_counts


proizvodi = list(transactions.columns)
proizvodi.remove('Cluster')



'''
Visualize Cluster Centers
'''

centers = k_means.cluster_centers_[2]


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
                              n_clusters=4)

model = clt.fit(test.todense())


transactions.columns


transactions['Cluster'] = model.labels_

cluster_counts = transactions['Cluster'].value_counts()

cluster_counts

'''
Print WordCloud
'''

cluster_selection = transactions['Cluster'] == 2


# From counts


agg = test[cluster_selection]

# From TF_IDF not Counts

agg = pd.DataFrame(agg[cluster_selection].todense())


agg.columns=proizvodi



#agg = agg.iloc[:, 1:-1].sum(axis=0)

agg = agg.mean(axis=0)

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

