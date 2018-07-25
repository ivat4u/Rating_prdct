import pandas as pd
import os
import time
import numpy as np

#load data into matrix
def load():
    path=os.getcwd()+'\ml-latest-small\movies.csv'
    data_movies= pd.read_csv(path, header=0, names=['movieId','title','genres'])

    path=os.getcwd()+'\ml-latest-small\\ratings.csv'
    data_ratings= pd.read_csv(path, header=0, names=['userId','movieId','rating','timestamp'])

    path=os.getcwd()+'\ml-latest-small\\tags.csv'
    data_tags= pd.read_csv(path, header=0, names=['userId','movieId','tag','timestamp'])
    path=os.getcwd()+'\ml-latest-small\\links.csv'
    data_links= pd.read_csv(path, header=0, names=['movieId','imdbId','tmdbId'])
    return data_ratings,data_movies,data_links,data_tags
data_ratings,data_movies,data_links,data_tags=load()

#get the different classes of movies,save as a dict(list)
def getclass():

    class_test=data_movies['genres'].str.split('|',expand=True)
    class_test=class_test.values.reshape((-1,1)).flatten()
    class_test=pd.Series(class_test)
    dict_class=pd.unique(class_test).tolist()
    return dict_class

dict = (getclass())
Num_class=len(dict)
# genres become sparse matrix
def getweight_table():

    table_weight = pd.DataFrame(data=data_movies.values, columns=('movieId', 'title', 'genres'))
    i=0
    for item in data_movies.values:
        '''dict_class = dict(key=tuple(getclass()))
        dict_class=dict_class.fromkeys(tuple(getclass()), '0')'''
        table_weight.loc[i][0]=item[0]
        table_weight.loc[i][1] = item[1]
        '''table=item[2].split('|')
        for str in table:
            if str!='key':
                dict_class['%s'%str]=1
        table_weight.loc[i][2] = dict_class'''
        genres = []
        for j in range(len(dict)):
            genres.append(0)
        table = item[2].split('|')
        for str in table:
            for s in dict:
                if str==s:
                    genres[dict.index(s)]=1
        table_weight.loc[i][2] = genres
        i += 1
    return table_weight
table_movies=getweight_table()

#Minus average for the rating table
def get_ratings_table():
    table_rating = pd.DataFrame(data=data_ratings.values, columns=('userId', 'movieId', 'rating','timestamp'))
    usersId=pd.unique(data_ratings['userId'])
    average=np.array(usersId,dtype=float)
    usersId=np.row_stack((usersId, average)).astype(float)
    for user in usersId[0]:
        usersId[1][np.where(usersId[0]==user)]=table_rating[table_rating.userId==user].rating.mean()
    for item in table_rating.values:
        item[2]=item[2]-usersId[1][np.where(usersId[0]==item[0])]
    return table_rating
table_rating=get_ratings_table()

def get_link_train_table():
    table_train=pd.DataFrame(table_rating.copy())
    table_train.astype(object)
    table_train.insert(0,'genres',list)
    table_train.insert(4, 'title','name')
    df = pd.DataFrame(columns=["genres", "userId", "movieId", "rating", "title", "timestamp"])
    i=0
    for item in table_train.values:
        item[0]=  table_movies[table_movies.movieId==int(item[2])].genres.values.real
        item[4] = table_movies[table_movies.movieId == int(item[2])].title.values.real

        df.loc[i] = item
        i=i+1
    table_train=df
    return  table_train
#table_train=get_link_train_table()
#table_train.to_csv('table_train.csv',)