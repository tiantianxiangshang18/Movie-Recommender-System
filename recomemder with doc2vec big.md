
Movie Recommender System to Give Personalized Recommendations

Built a hybrid movie recommender system using the metadata for 45,000 movies and ratings from 270,000 users.

Built a content-based recommender based on plot descriptions. Doc2Vec based on paragraph vector was applied to find movies with similar plot descriptions.

Combined content-based with collaborative filter-based engines to establish a hybrid movie recommender system to give personalized recommendations for different users.

This example shows the hybrid movie recommender system gives different recommendations to different users according to their previous rating for other movies.


```python
hybrid(1, 'The Shawshank Redemption')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26728</th>
      <td>The Last Gangster</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>1937</td>
      <td>28712</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>The Housekeeper</td>
      <td>8.0</td>
      <td>6.1</td>
      <td>2002</td>
      <td>34840</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>39876</th>
      <td>3 Geezers!</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>2013</td>
      <td>190967</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>27044</th>
      <td>Lifeguard</td>
      <td>5.0</td>
      <td>6.7</td>
      <td>1976</td>
      <td>5002</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>16051</th>
      <td>Undisputed III : Redemption</td>
      <td>182.0</td>
      <td>7.3</td>
      <td>2010</td>
      <td>38234</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>15847</th>
      <td>The Oscar</td>
      <td>3.0</td>
      <td>6.3</td>
      <td>1966</td>
      <td>56168</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>30409</th>
      <td>Curfew</td>
      <td>33.0</td>
      <td>7.5</td>
      <td>2012</td>
      <td>157289</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>16717</th>
      <td>Three Crowns of the Sailor</td>
      <td>7.0</td>
      <td>8.3</td>
      <td>1983</td>
      <td>64441</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>25336</th>
      <td>H6: Diary of a Serial Killer</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2005</td>
      <td>39928</td>
      <td>2.70381</td>
    </tr>
  </tbody>
</table>
</div>




```python
hybrid(50, 'The Shawshank Redemption')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26728</th>
      <td>The Last Gangster</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>1937</td>
      <td>28712</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>The Housekeeper</td>
      <td>8.0</td>
      <td>6.1</td>
      <td>2002</td>
      <td>34840</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>39876</th>
      <td>3 Geezers!</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>2013</td>
      <td>190967</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>27044</th>
      <td>Lifeguard</td>
      <td>5.0</td>
      <td>6.7</td>
      <td>1976</td>
      <td>5002</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>16051</th>
      <td>Undisputed III : Redemption</td>
      <td>182.0</td>
      <td>7.3</td>
      <td>2010</td>
      <td>38234</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>15847</th>
      <td>The Oscar</td>
      <td>3.0</td>
      <td>6.3</td>
      <td>1966</td>
      <td>56168</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>30409</th>
      <td>Curfew</td>
      <td>33.0</td>
      <td>7.5</td>
      <td>2012</td>
      <td>157289</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>16717</th>
      <td>Three Crowns of the Sailor</td>
      <td>7.0</td>
      <td>8.3</td>
      <td>1983</td>
      <td>64441</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>25336</th>
      <td>H6: Diary of a Serial Killer</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2005</td>
      <td>39928</td>
      <td>3.310722</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate
import gensim
from gensim.models.doc2vec import Doc2Vec
import warnings; warnings.simplefilter('ignore')
```


```python
os.chdir(r'C:/movie project')
```

Simple Recommender


```python
md = pd. read_csv('movies_metadata.csv')
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.95)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')

def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
```


```python
s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
```


```python
qualified.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>year</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>popularity</th>
      <th>genres</th>
      <th>wr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15480</th>
      <td>Inception</td>
      <td>2010</td>
      <td>14075</td>
      <td>8</td>
      <td>29.1081</td>
      <td>[Action, Thriller, Science Fiction, Mystery, A...</td>
      <td>7.917588</td>
    </tr>
    <tr>
      <th>12481</th>
      <td>The Dark Knight</td>
      <td>2008</td>
      <td>12269</td>
      <td>8</td>
      <td>123.167</td>
      <td>[Drama, Action, Crime, Thriller]</td>
      <td>7.905871</td>
    </tr>
    <tr>
      <th>22879</th>
      <td>Interstellar</td>
      <td>2014</td>
      <td>11187</td>
      <td>8</td>
      <td>32.2135</td>
      <td>[Adventure, Drama, Science Fiction]</td>
      <td>7.897107</td>
    </tr>
    <tr>
      <th>2843</th>
      <td>Fight Club</td>
      <td>1999</td>
      <td>9678</td>
      <td>8</td>
      <td>63.8696</td>
      <td>[Drama]</td>
      <td>7.881753</td>
    </tr>
    <tr>
      <th>4863</th>
      <td>The Lord of the Rings: The Fellowship of the Ring</td>
      <td>2001</td>
      <td>8892</td>
      <td>8</td>
      <td>32.0707</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.871787</td>
    </tr>
    <tr>
      <th>292</th>
      <td>Pulp Fiction</td>
      <td>1994</td>
      <td>8670</td>
      <td>8</td>
      <td>140.95</td>
      <td>[Thriller, Crime]</td>
      <td>7.868660</td>
    </tr>
    <tr>
      <th>314</th>
      <td>The Shawshank Redemption</td>
      <td>1994</td>
      <td>8358</td>
      <td>8</td>
      <td>51.6454</td>
      <td>[Drama, Crime]</td>
      <td>7.864000</td>
    </tr>
    <tr>
      <th>7000</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>2003</td>
      <td>8226</td>
      <td>8</td>
      <td>29.3244</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.861927</td>
    </tr>
    <tr>
      <th>351</th>
      <td>Forrest Gump</td>
      <td>1994</td>
      <td>8147</td>
      <td>8</td>
      <td>48.3072</td>
      <td>[Comedy, Drama, Romance]</td>
      <td>7.860656</td>
    </tr>
    <tr>
      <th>5814</th>
      <td>The Lord of the Rings: The Two Towers</td>
      <td>2002</td>
      <td>7641</td>
      <td>8</td>
      <td>29.4235</td>
      <td>[Adventure, Fantasy, Action]</td>
      <td>7.851924</td>
    </tr>
  </tbody>
</table>
</div>



Content Based Recommender

Read csv file and delete 19730, 29503, 35587


```python
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
```


```python
smd = md
smd['description'] = smd['overview'].fillna('')
smd['description'] = smd['description'].astype('str').apply(lambda x: str.lower(x.replace(",", "")))
smd['description'] = smd['description'].astype('str').apply(lambda x: str.lower(x.replace(".", "")))
```


```python
def getText():
    discuss_train=list(smd['description'])
    return discuss_train
 
text=getText()

TaggededDocument=gensim.models.doc2vec.TaggedDocument

def X_train(cut_sentence):
    x_train=[]
    for i, text in enumerate(cut_sentence):
        word_list=text.split(' ')
        l=len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document=TaggededDocument(word_list,tags=[i])
        x_train.append(document)
    return x_train

c=X_train(text)

def train(x_train):
    model=Doc2Vec(x_train, min_count=1, size=200)
    return model

model_dm=train(c)

md = md.reset_index()
titles = md['title']
indices = pd.Series(md.index, index=md['title'])

```


```python
def get_recommendations(title):
    idx = indices[title]
    strl= md['overview'].iloc[idx]
    test_text=strl.split(' ')
    inferred_vector=model_dm.infer_vector(doc_words=test_text,alpha=0.025, min_alpha = 0.001, steps=10000)
    sims = model_dm.docvecs.most_similar([inferred_vector],topn=11)
    movie_indices = [i[0] for i in sims[1:10]]
    return titles.iloc[movie_indices]
```

Check the recommendations for The Shawshank Redemption


```python
get_recommendations('The Shawshank Redemption')
```




    24186                           The Uncertainty Principle
    5443                                             Quitting
    20581                                    The Great Gatsby
    42936                                  Unexpected Journey
    32789                                           12 Chairs
    25888    Johan Falk: GSI - Gruppen för särskilda insatser
    1216                                         Evil Dead II
    22654                     Better Living Through Chemistry
    11773                                  The Devil Commands
    Name: title, dtype: object



Popularity and Ratings


```python
smd = md
smd['description'] = smd['overview'].fillna('')
def improved_recommendations(title):
    idx = indices[title]
    strl= smd['description'].iloc[idx]
    test_text=strl.split(' ')
    inferred_vector=model_dm.infer_vector(doc_words=test_text,alpha=0.025, min_alpha = 0.001, steps=10000)
    sims = model_dm.docvecs.most_similar([inferred_vector],topn=50)
    movie_indices = [i[0] for i in sims[1:50]]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
```


```python
improved_recommendations('The Shawshank Redemption')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>wr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43641</th>
      <td>Baby Driver</td>
      <td>2083</td>
      <td>7</td>
      <td>2017</td>
      <td>6.697372</td>
    </tr>
    <tr>
      <th>23169</th>
      <td>The Raid 2</td>
      <td>832</td>
      <td>7</td>
      <td>2014</td>
      <td>6.398329</td>
    </tr>
    <tr>
      <th>10332</th>
      <td>Transporter 2</td>
      <td>1076</td>
      <td>6</td>
      <td>2005</td>
      <td>5.782970</td>
    </tr>
    <tr>
      <th>16051</th>
      <td>Undisputed III : Redemption</td>
      <td>182</td>
      <td>7</td>
      <td>2010</td>
      <td>5.763450</td>
    </tr>
    <tr>
      <th>10369</th>
      <td>Domino</td>
      <td>450</td>
      <td>6</td>
      <td>2005</td>
      <td>5.629282</td>
    </tr>
    <tr>
      <th>10919</th>
      <td>Magic</td>
      <td>59</td>
      <td>7</td>
      <td>1978</td>
      <td>5.454939</td>
    </tr>
    <tr>
      <th>9086</th>
      <td>Pusher</td>
      <td>162</td>
      <td>6</td>
      <td>1996</td>
      <td>5.450143</td>
    </tr>
    <tr>
      <th>30409</th>
      <td>Curfew</td>
      <td>33</td>
      <td>7</td>
      <td>2012</td>
      <td>5.368919</td>
    </tr>
    <tr>
      <th>6424</th>
      <td>Avanti!</td>
      <td>49</td>
      <td>6</td>
      <td>1972</td>
      <td>5.321501</td>
    </tr>
    <tr>
      <th>18839</th>
      <td>Madhouse</td>
      <td>21</td>
      <td>6</td>
      <td>1974</td>
      <td>5.279748</td>
    </tr>
  </tbody>
</table>
</div>



Collaborative Filtering


```python
reader = Reader()
```


```python
ratings = pd.read_csv('ratings_small.csv')
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=5)
```


```python
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
```

    Evaluating RMSE, MAE of algorithm SVD.
    
    ------------
    Fold 1
    RMSE: 0.8907
    MAE:  0.6876
    ------------
    Fold 2
    RMSE: 0.8920
    MAE:  0.6853
    ------------
    Fold 3
    RMSE: 0.9027
    MAE:  0.6968
    ------------
    Fold 4
    RMSE: 0.9049
    MAE:  0.6984
    ------------
    Fold 5
    RMSE: 0.8990
    MAE:  0.6905
    ------------
    ------------
    Mean RMSE: 0.8979
    Mean MAE : 0.6917
    ------------
    ------------
    




    CaseInsensitiveDefaultDict(list,
                               {'mae': [0.6876289845916422,
                                 0.6853187852842678,
                                 0.6968463315260166,
                                 0.6983666668898275,
                                 0.6905333288418136],
                                'rmse': [0.8907329120066598,
                                 0.8919991071163405,
                                 0.9027361586391015,
                                 0.9048594539655962,
                                 0.8990288481051337]})




```python
trainset = data.build_full_trainset()
svd.train(trainset)
```




    <surprise.prediction_algorithms.matrix_factorization.SVD at 0x1a286175a58>




```python
ratings[ratings['userId'] == 1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>31</td>
      <td>2.5</td>
      <td>1260759144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1029</td>
      <td>3.0</td>
      <td>1260759179</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1061</td>
      <td>3.0</td>
      <td>1260759182</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1129</td>
      <td>2.0</td>
      <td>1260759185</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1172</td>
      <td>4.0</td>
      <td>1260759205</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1263</td>
      <td>2.0</td>
      <td>1260759151</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1287</td>
      <td>2.0</td>
      <td>1260759187</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1293</td>
      <td>2.0</td>
      <td>1260759148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>1339</td>
      <td>3.5</td>
      <td>1260759125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1343</td>
      <td>2.0</td>
      <td>1260759131</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1371</td>
      <td>2.5</td>
      <td>1260759135</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>1405</td>
      <td>1.0</td>
      <td>1260759203</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1953</td>
      <td>4.0</td>
      <td>1260759191</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>2105</td>
      <td>4.0</td>
      <td>1260759139</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>2150</td>
      <td>3.0</td>
      <td>1260759194</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>2193</td>
      <td>2.0</td>
      <td>1260759198</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>2294</td>
      <td>2.0</td>
      <td>1260759108</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>2455</td>
      <td>2.5</td>
      <td>1260759113</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>2968</td>
      <td>1.0</td>
      <td>1260759200</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>3671</td>
      <td>3.0</td>
      <td>1260759117</td>
    </tr>
  </tbody>
</table>
</div>




```python
svd.predict(1, 302, 3)
```




    Prediction(uid=1, iid=302, r_ui=3, est=2.7111648412283342, details={'was_impossible': False})



Hybrid Recommender


```python
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
```


```python
#new link
id_map = pd.read_csv('links.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
```


```python
id_map = pd.read_csv('links.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
```


```python
indices_map = id_map.set_index('id')
```


```python
def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    strl= smd['description'].iloc[idx]
    test_text=strl.split(' ')
    inferred_vector=model_dm.infer_vector(doc_words=test_text,alpha=0.025, min_alpha = 0.001, steps=10000)
    sims = model_dm.docvecs.most_similar([inferred_vector])
    movie_indices = [i[0] for i in sims[1:50]]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
```


```python
hybrid(1, 'The Shawshank Redemption')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26728</th>
      <td>The Last Gangster</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>1937</td>
      <td>28712</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>The Housekeeper</td>
      <td>8.0</td>
      <td>6.1</td>
      <td>2002</td>
      <td>34840</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>39876</th>
      <td>3 Geezers!</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>2013</td>
      <td>190967</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>27044</th>
      <td>Lifeguard</td>
      <td>5.0</td>
      <td>6.7</td>
      <td>1976</td>
      <td>5002</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>16051</th>
      <td>Undisputed III : Redemption</td>
      <td>182.0</td>
      <td>7.3</td>
      <td>2010</td>
      <td>38234</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>15847</th>
      <td>The Oscar</td>
      <td>3.0</td>
      <td>6.3</td>
      <td>1966</td>
      <td>56168</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>30409</th>
      <td>Curfew</td>
      <td>33.0</td>
      <td>7.5</td>
      <td>2012</td>
      <td>157289</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>16717</th>
      <td>Three Crowns of the Sailor</td>
      <td>7.0</td>
      <td>8.3</td>
      <td>1983</td>
      <td>64441</td>
      <td>2.70381</td>
    </tr>
    <tr>
      <th>25336</th>
      <td>H6: Diary of a Serial Killer</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2005</td>
      <td>39928</td>
      <td>2.70381</td>
    </tr>
  </tbody>
</table>
</div>




```python
hybrid(50, 'The Shawshank Redemption')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26728</th>
      <td>The Last Gangster</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>1937</td>
      <td>28712</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>The Housekeeper</td>
      <td>8.0</td>
      <td>6.1</td>
      <td>2002</td>
      <td>34840</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>39876</th>
      <td>3 Geezers!</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>2013</td>
      <td>190967</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>16051</th>
      <td>Undisputed III : Redemption</td>
      <td>182.0</td>
      <td>7.3</td>
      <td>2010</td>
      <td>38234</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>27044</th>
      <td>Lifeguard</td>
      <td>5.0</td>
      <td>6.7</td>
      <td>1976</td>
      <td>5002</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>15847</th>
      <td>The Oscar</td>
      <td>3.0</td>
      <td>6.3</td>
      <td>1966</td>
      <td>56168</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>30409</th>
      <td>Curfew</td>
      <td>33.0</td>
      <td>7.5</td>
      <td>2012</td>
      <td>157289</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>18624</th>
      <td>Miss Nobody</td>
      <td>15.0</td>
      <td>4.3</td>
      <td>2010</td>
      <td>51875</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>25336</th>
      <td>H6: Diary of a Serial Killer</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>2005</td>
      <td>39928</td>
      <td>3.310722</td>
    </tr>
  </tbody>
</table>
</div>




```python
hybrid(1, 'The Godfather')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>700</th>
      <td>Dead Man</td>
      <td>397.0</td>
      <td>7.2</td>
      <td>1995</td>
      <td>922</td>
      <td>2.820908</td>
    </tr>
    <tr>
      <th>29532</th>
      <td>Mother's Day</td>
      <td>126.0</td>
      <td>6.3</td>
      <td>2010</td>
      <td>101669</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>24296</th>
      <td>Charlie Chan at the Opera</td>
      <td>14.0</td>
      <td>6.6</td>
      <td>1936</td>
      <td>28044</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>39342</th>
      <td>Les Tuche 2: Le rêve américain</td>
      <td>239.0</td>
      <td>5.7</td>
      <td>2016</td>
      <td>369776</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>39623</th>
      <td>Batman: The Killing Joke</td>
      <td>485.0</td>
      <td>6.2</td>
      <td>2016</td>
      <td>382322</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>30725</th>
      <td>Hallettsville</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2009</td>
      <td>9935</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>38167</th>
      <td>Moonshine County Express</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1977</td>
      <td>99846</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>10352</th>
      <td>Bookies</td>
      <td>19.0</td>
      <td>6.8</td>
      <td>2003</td>
      <td>14759</td>
      <td>2.703810</td>
    </tr>
    <tr>
      <th>37019</th>
      <td>Eve's Christmas</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>2004</td>
      <td>69016</td>
      <td>2.703810</td>
    </tr>
  </tbody>
</table>
</div>




```python
hybrid(50, 'The Godfather')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>year</th>
      <th>id</th>
      <th>est</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29532</th>
      <td>Mother's Day</td>
      <td>126.0</td>
      <td>6.3</td>
      <td>2010</td>
      <td>101669</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>39342</th>
      <td>Les Tuche 2: Le rêve américain</td>
      <td>239.0</td>
      <td>5.7</td>
      <td>2016</td>
      <td>369776</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>24296</th>
      <td>Charlie Chan at the Opera</td>
      <td>14.0</td>
      <td>6.6</td>
      <td>1936</td>
      <td>28044</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>39623</th>
      <td>Batman: The Killing Joke</td>
      <td>485.0</td>
      <td>6.2</td>
      <td>2016</td>
      <td>382322</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>30725</th>
      <td>Hallettsville</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>2009</td>
      <td>9935</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>38167</th>
      <td>Moonshine County Express</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1977</td>
      <td>99846</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>10352</th>
      <td>Bookies</td>
      <td>19.0</td>
      <td>6.8</td>
      <td>2003</td>
      <td>14759</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>44685</th>
      <td>Hurdy-Gurdy Hare</td>
      <td>2.0</td>
      <td>6.5</td>
      <td>1950</td>
      <td>236112</td>
      <td>3.310722</td>
    </tr>
    <tr>
      <th>700</th>
      <td>Dead Man</td>
      <td>397.0</td>
      <td>7.2</td>
      <td>1995</td>
      <td>922</td>
      <td>3.204096</td>
    </tr>
  </tbody>
</table>
</div>

