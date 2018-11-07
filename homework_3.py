
# coding: utf-8

# # CSCE 470 :: Information Storage and Retrieval :: Texas A&M University :: Fall 2018
# 
# 
# # Homework 3 (and 4):  Recommenders
# 
# ### 100 points [10% of your final grade; that's double!]
# 
# ### Due: November 8, 2018
# 
# *Goals of this homework:* Put your knowledge of recommenders to work. 
# 
# *Submission Instructions (Google Classroom):* To submit your homework, rename this notebook as  `lastname_firstinitial_hw#.ipynb`. For example, my homework submission would be: `caverlee_j_hw3.ipynb`. Submit this notebook via **Google Classroom**. Your IPython notebook should be completely self-contained, with the results visible in the notebook. We should not have to run any code from the command line, nor should we have to run your code within the notebook (though we reserve the right to do so).

# # Part 0: Movielens Data
# 
# For this first part, we're going to use part of the Movielens 100k dataset. Prior to the Netflix Prize, the Movielens data was **the** most important collection of movie ratings.
# 
# First off, we need to load the data (including u.user, u.item, and ua.base). Here, we provide you with some helper code to load the data using [Pandas](http://pandas.pydata.org/). Pandas is a nice package for Python data analytics.
# 
# You may need to install pandas doing something like:
# 
# `conda install --name cs470 pandas`

# In[1]:


import pandas as pd

# Load the user data
users_df = pd.read_csv('u.user', sep='|', names=['UserId', 'Age', 'Gender', 'Occupation', 'ZipCode'])

# Load the movies data: we will only use movie id and title for this homework
movies_df = pd.read_csv('u.item', sep='|', names=['MovieId', 'Title'], usecols=range(2), encoding = "ISO-8859-1")

# Load the ratings data: ignore the timestamps
ratings_df = pd.read_csv('ua.base', sep='\t', names=['UserId', 'MovieId', 'Rating'],usecols=range(3))

# Working on three different data frames is a pain
# Let us create a single dataset by "joining" these three data frames
movie_ratings_df = pd.merge(movies_df, ratings_df)
movielens_df = pd.merge(movie_ratings_df, users_df)

movielens_df.head()


# # Part 1. Let's find similar users [20 points]
# 
# Before we get to the actual task of building our recommender, let's familiarize ourselves with the Movielens data.
# 
# Pandas is really nice, since it let's us do simple aggregates. For example, we can find a specific user and take a look at that user's ratings. For example, for the user with user ID = 363, we have:

# In[2]:


gb = movielens_df.groupby('UserId')
User363 = gb.get_group(363)
#the information for the user
User363[:1][["UserId", "Age", "Gender","Occupation", "ZipCode"]]


# In[3]:


# And then we can see his first 10 ratings:
User363[['Title', 'Rating']][:10]


# Balderdash! Everyone agrees that Toy Story should be rated 5! Oh well, there's no accounting for taste.
# 
# Moving on, let's try our hand at finding similar users to this base user (user ID = 363). In each of the following, **find the top-10 most similar users** to this base user. You should use all of the user's ratings, not just the top-10 like we showed above. We're going to try different similarity methods and see what differences arise.
# 
# You should implement each of these similar methods yourself! 
# 
# ###     Top-10 Most Similar Users Using
# #### Jaccard

# In[4]:


import numpy as np
#setup ratings matrix

def getRatingMatrix(ratingsDf):
    matrix = np.zeros([ratingsDf['UserId'].max()+1, ratingsDf['MovieId'].max()+1])
    print(ratingsDf['UserId'].max()+1, ratingsDf['MovieId'].max()+1)
    gb = ratingsDf.groupby('UserId')
    for userId, group in gb:
        for mindex, mRow in group.iterrows():
            movieId = mRow['MovieId']
            matrix[userId][movieId] = int(mRow['Rating'])
    return matrix


# In[5]:


ratingMatrix= getRatingMatrix(movielens_df)
ratingMatrixT = ratingMatrix.transpose()


# In[6]:


def jaccard(user1Id, user2Id, matrix):
    u1Vec = np.where(matrix[user1Id] > 0, 1, 0)
    u2Vec = np.where(matrix[user2Id] > 0, 1, 0)
    
    intersection = (np.dot(u1Vec, u2Vec))
    A = np.sum(u1Vec)
    B = np.sum(u2Vec)
    return intersection / (A + B - intersection) 

users_df_copy = users_df
jaccard_df = users_df_copy.apply(lambda row: pd.Series(
    {
        'UserId': row['UserId'],
        'jaccard': jaccard (363, row['UserId'], ratingMatrix)
    }),axis=1)
jaccard_df.sort_values('jaccard', ascending = False)[1:11]


# ####     Cosine

# In[7]:


import math

def normalize(vec):
    magnitude = np.linalg.norm(vec)
    return vec / magnitude

def magnitude(vec):
    return np.linalg.norm(vec)

def cosine(user1Id, user2Id, matrix):
    u1Vec = matrix[user1Id]
    ANorm = magnitude(u1Vec)
    u2Vec = matrix[user2Id]
    BNorm = magnitude(u2Vec)
    dot = np.dot(u1Vec, u2Vec)
    return (dot / (ANorm*BNorm))
    
users_df_copy = users_df
users_df_copy = users_df_copy[users_df_copy.UserId != 363]

cosine_df = users_df_copy.apply(lambda row: pd.Series(
    {
        'UserId': row['UserId'],
        'cosine': cosine(363,row['UserId'], ratingMatrix)
    }),axis=1)
cosine_df.sort_values('cosine', ascending = False)[0:10]


# #### Pearson

# In[8]:


#setup the intersection we'll use in the next part
#this seems a little more complicated than it needs to be, but its really slow otherwise
def intersection(user1Id, user2Id, matrix):
    #convert arrays to 1 or 0
    u1Vec = np.where(matrix[user1Id] > 0, 1, 0)
    u2Vec = np.where(matrix[user2Id] > 0, 1, 0)
    
    #find which elements both have 1's
    intersect = np.logical_and(u1Vec, u2Vec)
    
    #find all values that were 0 in the intersect and make them 0 
    u1Intersect = intersect * matrix[user1Id]
    u2Intersect = intersect * matrix[user2Id]
    
    #remove all the zeros
    return (np.extract(u1Intersect > 0, u1Intersect), np.extract(u2Intersect > 0, u2Intersect))


# In[9]:


import math
import time

def pearson(user1Id, user2Id, matrix):
    '''
    This is the pearson from the slides
    '''
    (u1Vec, u2Vec) = intersection(user1Id, user2Id, matrix)
    u1Mean = np.mean(np.extract(matrix[user1Id] > 0, matrix[user1Id]))
    u2Mean = np.mean(np.extract(matrix[user2Id] > 0, matrix[user2Id]))
    u1Diff = (u1Vec - u1Mean)
    u2Diff = (u2Vec - u2Mean)
    covariance = (u1Diff * u2Diff).sum()
    bottom = math.sqrt((u1Diff*u1Diff).sum()*(u2Diff*u2Diff).sum())
    return covariance / bottom

users_df_copy = users_df
users_df_copy = users_df_copy[users_df_copy.UserId != 363]

pearson_df = users_df_copy.apply(lambda row: pd.Series(
    {
        'UserId': row['UserId'],
        'pearson': pearson (363, row['UserId'], ratingMatrix)
    }),axis=1)

pearson_df.sort_values('pearson', ascending = False)[0:10]


# ### What are the differences among these three similarity methods? Which one do you prefer, why?

#  Jaccard does not care about the rating that the user gave, only that it is included in the set of movies they rated, This causes it to maybe not be the best because it doesent take into account whether a user actually liked the movie. Cosine is better because it takes into account the rating, but it counts all of the movies that a user hasent rated but the other has as a 0, this may not be an accurate representation of a users score for that movie since they may have just not seen it yet. Pearson I think is the best because it takes into account ratings, and doesn't assume as much as cosine, because it only looks at movies that both have rated.

# # Part 2: User-User Collaborative Filtering: Similarity-Based Ratings Prediction [20 points]
# 
# Now let's estimate the rating of UserID 363 for the movie "Dances with Wolves (1990)" (MovieId 97) based on the similar users. Find the 10 nearest (most similiar by using Pearson) users who rated the movie "Dances with Wolves (1990)" and try different aggregate functions. Recall, there are many different ways to aggregate the ratings of the nearest neighbors. We'll try three popular methods:
# 
# ### Method 1: Average. 
# The first is to simply average the ratings of the nearest neighbors:
# $r_{c,s} = \frac{1}{N}\sum_{c'\in \hat{C}}r_{c',s}$

# In[10]:


def getClosestkSimilarUsers(movieId, userId, similarityFunc, k, matrix, matrixT):
    usersWhoSawMovie = np.where(matrixT[movieId] > 0, 1, 0)
    #userIndexes is just [1,2,3,...,n]
    userIndexes = np.fromfunction(lambda i,j: j, (1,len(usersWhoSawMovie)), dtype=int)[0]
    usersWhoSawMovieIndexes = (usersWhoSawMovie * userIndexes)
    usersWhoSawMovieIndexesFiltered = np.extract(usersWhoSawMovieIndexes > 0, usersWhoSawMovieIndexes)
    
    usersWithSim = (np.array(list(
        map(lambda item2: 
            (item2, similarityFunc (userId, item2, matrix), matrix[item2][movieId]),
             usersWhoSawMovieIndexesFiltered))))
    usersWithSim = np.nan_to_num(usersWithSim,0)
    if(len(usersWithSim) is 0):
        print(movieId, userId)
        return pd.DataFrame(data = [[0,0,0]], columns= ['UserId', 'similarity', 'Rating'])
    sortedTopK = usersWithSim[usersWithSim[:,1].argsort()[::-1]][:k]
    return pd.DataFrame(data = sortedTopK, columns= ['UserId', 'similarity', 'Rating'])

print('predicted score = ', getClosestkSimilarUsers(97,363, pearson, 10, ratingMatrix, ratingMatrixT)['Rating'].mean())


# ### Method 2: Weighted Average 1. 
# The second is to take a weighted average, where we weight more "similar" neighbors higher: $r_{c,s} = k\sum_{c'\in \hat{C}}sim(c, c')\times r_{c',s}$
# 
# Choose a reasonable k so that r_{c,s} is between 1 to 5

# In[11]:


def weightedAverage(movieId, userId, similarityFunc, k, matrix, matrixT):
    closestk = getClosestkSimilarUsers(movieId,userId, similarityFunc, k, matrix, matrixT)
    k = 1/closestk['similarity'].sum()
    score = closestk.apply(lambda row: k * row['similarity'] * row['Rating'], axis=1).sum()
    return score

print('predicted score = ', weightedAverage(97, 363, pearson, 10, ratingMatrix, ratingMatrixT))


# ### Method 3: Weighted Average 2. 
# An alternative weighted average is to weight the differences between their ratings and their average rating (in essence to reward movies that are above the mean): $r_{c,s} = \bar{r}_c + k\sum_{c'\in \hat{C}}sim(c, c')\times (r_{c',s} - \bar{r}_{c'})$
# 
# Choose a reasonable k so that r_{c,s} is between 1 to 5

# In[12]:


def weightedAverage2(movieId, userId, similarityFunc, k, matrix,matrixT):
    closestk = getClosestkSimilarUsers(movieId,userId, similarityFunc, k, matrix, matrixT)
    k = 1/closestk['similarity'].sum()
    rMean = np.mean(np.extract(matrix[userId] > 0, matrix[userId]))
    def innerSum(row):
        rUserId = int(row['UserId'])
        rOtherMean = np.mean(np.extract(matrix[rUserId] > 0, matrix[rUserId]))
        return row['similarity'] * (row['Rating'] - rOtherMean)
    score = rMean + k * closestk.apply(innerSum, axis=1).sum()

    return score

print('predicted score = ', weightedAverage2(97, 363, pearson, 10, ratingMatrix, ratingMatrixT))


# # Part 3: Baseline Recommendation (Global) [20 points]
# 
# OK, so far we've built the basics of a user-user collaborative filtering approach; that is, we take a user, find similar users and then aggregate their ratings. 
# 
# An alternative approach is to consider just basic statistics of the movies and users themselves. This is the essence of the "baseline" recommender we discussed in class:
# 
# $b_{xi} = \mu + b_x + b_i$
# 
# where $b_{x,i}$ is the baseline estimate rating user x would give to item i, $\mu$ is the overall mean rating, $b_x$ is a user bias term, and $b_i$ is an item bias term.
# 
# For this part, let's once again estimate the rating of UserID 363 for the movie "Dances with Wolves (1990)" (MovieId 97), but this time using the baseline recommender.

# In[13]:


def getBaseline(userId, itemId, matrix, matrixT, avgGlobal):
    #we ask them to give us the global because it takes a long time to calculate
    start = time.time()
    #avgGlobal = np.extract(matrix > 0, matrix).mean()
    avgUser = np.extract(matrix[userId] > 0, matrix[userId]).mean()
    avgMovie = np.extract(matrixT[itemId] > 0, matrixT[itemId]).mean()
    end = time.time()
    #print(end - start)
    #print(avgGlobal + (avgUser - avgGlobal) + (avgMovie - avgGlobal))
    return avgGlobal + (avgUser - avgGlobal) + (avgMovie - avgGlobal)

avgGlobal = np.extract(ratingMatrix > 0, ratingMatrix).mean()
print('baseline for user 363, item 97 = ', getBaseline(363, 97, ratingMatrix, ratingMatrixT, avgGlobal))
#avgGlobal = np.extract(trainingRatingMatrix > 0, trainingRatingMatrix).mean()
#getBaseline(405, 1582, trainingRatingMatrix, trainingRatingMatrixT, avgGlobal)


# # Part 4: Local + Global Recommendation (Baseline + Item-Item CF) [20 points]
# 
# Our final recommender combines the global baseline recommender with an item-item local recommender. 
# 
# $\hat{r}_{xi} = b_{xi} + \dfrac{\sum_{j \in N(i;x)}s_{ij} \cdot (r_{xj} - b_{xj})}{\sum_{j \in N(i;x)}s_{ij}} $
# 
# where 
# * $\hat{r}_{xi}$ is our estimated rating for what user x would give to item i.
# * $s_{ij}$ is the similarity of items i and j.
# * $r_{xj}$ is the rating of user x on item j.
# * $N(i;x)$ is the set of items similar to item i that were rated by x.
# 
# You will need to make some design choices here about what similarity measure to use and what threshold to determine what items belong in $N(i;x)$.
# 
# Now show us what this estimates the rating of UserID 363 for the movie "Dances with Wolves (1990)" (MovieId 97) to be:

# In[14]:


def itemItemSim(userId, movieId, matrix, matrixT, similarityFunc):
    #get the movies the user has seen as an array of 1s and 0s
    moviesUserHasSeen = np.where(matrix[userId] > 0, 1, 0)
    
    #get an array of just indexes for the movies
    movieIndexes = np.fromfunction(lambda i,j: j, (1,len(matrix[userId])), dtype=int)[0]
    #change the 1s to be the index of the movie it represents, and filter 0s
    moviesUserHasSeenIndexes = (moviesUserHasSeen * movieIndexes)
    
    moviesUserHasSeenFiltered = np.extract(moviesUserHasSeenIndexes > 0, moviesUserHasSeenIndexes)
    
    moviesWithSim = (np.array(list(map
             (
                lambda otherMovieId:
                 (
                    otherMovieId,
                    similarityFunc(movieId, otherMovieId, matrixT),
                    matrix[userId][otherMovieId]
                ),
                moviesUserHasSeenFiltered
             ))))
    moviesWithSim = np.nan_to_num(moviesWithSim,0)
    
    return moviesWithSim

def itemItemAboveCutoff(userId, movieId, cutoff, matrix, matrixT, similarityFunc):
    moviesWithSim = itemItemSim(userId, movieId, matrix, matrixT, similarityFunc)
    sortedSim = moviesWithSim[moviesWithSim[:,1].argsort()[::-1]]
    aboveCutoff = sortedSim[sortedSim[:,1] > cutoff]
    return aboveCutoff

def itemItemClosestK(userId, movieId, k, matrix, matrixT, similarityFunc):
    moviesWithSim = itemItemSim(userId, movieId, matrix, matrixT, similarityFunc)
    sortedTopK = moviesWithSim[moviesWithSim[:,1].argsort()[::-1]][:k]
    
    return pd.DataFrame(data = sortedTopK, columns= ['MovieId', 'similarity', 'Rating'])

ratingMatrixT = ratingMatrix.transpose()
#itemItemSim(363, 97, ratingMatrix, ratingMatrixT, jaccard)
#itemItemClosestK(363, 97, 10, ratingMatrix, ratingMatrixT, jaccard)
#itemItemAboveCutoff(363, 97, .4, ratingMatrix, ratingMatrixT, jaccard)


# In[15]:


def localRating(userId, movieId, ratingsDf, matrix, matrixT, avgGlobal, k, similarityFunc):
    start = time.time()
    gb = ratingsDf.groupby('UserId')
    top = 0
    bot = 0
    for mIndex, mRow in itemItemClosestK(userId, movieId, k, matrix, matrixT, similarityFunc).iterrows():
        sim = mRow['similarity']
        bot += sim
        rating = int(mRow['Rating'])
        base = getBaseline(userId, int(mRow['MovieId']), matrix, matrixT, avgGlobal)
        top += sim * (rating - base)
    print(time.time() - start)
    return (top / bot)

avgGlobal = np.extract(ratingMatrix > 0, ratingMatrix).mean()

print(getBaseline(363, 97, ratingMatrix, ratingMatrixT, avgGlobal) + 
      localRating(363, 97, ratings_df, ratingMatrix, ratingMatrixT, avgGlobal, 10, cosine))


# In[16]:


def localRatingCutoff(userId, movieId, ratingsDf, matrix, matrixT, avgGlobal, cutoff, similarityFunc):
    top = 0
    bot = 0
    simItems = itemItemAboveCutoff(userId, movieId, cutoff, matrix, matrixT, similarityFunc)
    def topCalc(row):
        rowMovieId = row[0]
        sim = row[1]
        rating = row[2]
        baseline = getBaseline(userId, int(rowMovieId), matrix, matrixT, avgGlobal)
        return sim * (rating - baseline)
    if(len(simItems) is 0):
        return 0
    top = np.apply_along_axis(topCalc, 1, simItems).sum()
    bot = simItems[:,1].sum()
    if(bot is 0):
        return 0
    
    return (top / bot)

print(getBaseline(363, 97, ratingMatrix, ratingMatrixT, avgGlobal) + 
      localRatingCutoff(363, 97, ratings_df, ratingMatrix, ratingMatrixT, avgGlobal, .5, jaccard))


# # Part 5. Putting it all together! [20 points]
# 
# Finally, we're going to experiment with our different methods to see which one performs the best on our special test set of "hidden" ratings. We have three big "kinds" of recommenders:
# 
# * User-User Collaborative Filtering
# * Baseline Recommendation (Global)
# * Local + Global Recommender
# 
# 
# But within each, we have lots of design choices. For example, do we try Jaccard+Average or Jaccard+WeightedAverage1? Do we try Jaccard or Cosine or Pearson? What choice of k? Etc.
# 
# For this part, you should train your methods on a special train set (the base set, see below). Then report your results over the test set (see below). You should use RMSE as your metric of choice. Which method performs best? You will need to experiment with many different approaches, but ultimately, you should tell us the best 2 or 3 methods and report the RMSE you get.

# In[17]:


train = pd.read_csv('ua.base', sep='\t', names=['UserId', 'MovieId', 'Rating'],usecols=range(3))
test = pd.read_csv('ua.test', sep='\t', names=['UserId', 'MovieId', 'Rating'],usecols=range(3))


# In[18]:


trainingRatingMatrix= getRatingMatrix(train)
trainingRatingMatrixT = trainingRatingMatrix.transpose()


# In[19]:


import time
import math
def GetRMSE(testDf, predictionFunction):
    rmse = 0
    cnt = 0
    for index, row in testDf.iterrows():
        start = time.time()
        score = predictionFunction(row)
        if(math.isnan(score)):
            score = 0
        sqrDiff = math.pow(row['Rating'] - score, 2)
        rmse += sqrDiff
        cnt+=1
    return math.sqrt(rmse/cnt)


# In[20]:


#baseline
avgGlobal = np.extract(trainingRatingMatrix > 0, trainingRatingMatrix).mean()
def baselinePrediction(row):
    return getBaseline(row['UserId'], row['MovieId'], trainingRatingMatrix, trainingRatingMatrixT, avgGlobal)

GetRMSE(test, baselinePrediction)


# In[21]:


def GetRMSE(testDf, predictionFunction, k, simFunc, simFuncName):
    rmse = 0
    cnt = 0
    for index, row in testDf.iterrows():
        start = time.time()
        score = predictionFunction(row, k, simFunc)
        if(math.isnan(score)):
            score = 0
        sqrDiff = math.pow(row['Rating'] - score, 2)
        #if(sqrDiff > 2):
        #    print(row, score)
        rmse += sqrDiff
        cnt+=1
        
    return {'k': k, 'similarityFunc': simFuncName, 'error' : math.sqrt(rmse/cnt)}


# In[22]:


from joblib import Parallel, delayed
import multiprocessing

num_cores = 4#multiprocessing.cpu_count()

def ParallelRMSE(testDf, predictionFunction, predictionFunctionName, ks, simFuncs):
    allPairs = [ (k,name,func) for k in ks for name, func in simFuncs.items()]
    results = Parallel(n_jobs=num_cores)(delayed(GetRMSE)(testDf,predictionFunction,k,func, name) for (k,name,func) in allPairs)
    error_df = pd.DataFrame(data=results)
    error_df['predictionFunction'] = predictionFunctionName
    return error_df


# In[23]:


def basicAverageUserUser(row, k, simFunc):
    return getClosestkSimilarUsers(
        row['MovieId'],
        row['UserId'],
        simFunc, 
        k, 
        trainingRatingMatrix, 
        trainingRatingMatrixT)['Rating'].mean()
ks = [5,10,15,25]
similarityFuncs = {'jaccard' : jaccard,'cosine' : cosine,'pearson' : pearson}


# In[25]:


ParallelRMSE(test[:1], basicAverageUserUser, "basicAverageUserUser", ks, similarityFuncs)   


# In[ ]:


#user user
def userUserWeightedAverage1Prediction(row, k, simFunc):
    #if we havent observed any rankings for this guy return global avg
    if(len(train[train.MovieId == row['MovieId']]) == 0):
        return avgGlobal
    score = weightedAverage(row['MovieId'], row['UserId'], simFunc, k, trainingRatingMatrix, trainingRatingMatrixT)
    return score


# In[ ]:


ParallelRMSE(test, userUserWeightedAverage1Prediction, "userUserWeightedAverage1Prediction", ks, similarityFuncs)


# In[ ]:


#user user2
def userUserWeightedAverage2Prediction(row, k, simFunc):
    #if we havent observed any rankings for this guy return global avg
    if(len(train[train.MovieId == row['MovieId']]) == 0):
        return avgGlobal
    return weightedAverage2(row['MovieId'], row['UserId'], simFunc, k,trainingRatingMatrix, trainingRatingMatrixT)


# In[ ]:


ParallelRMSE(test, userUserWeightedAverage2Prediction, "userUserWeightedAverage2Prediction", ks, similarityFuncs)


# In[ ]:


#local + global
def localPlusGlobalPrediction(row, k, simFunc):
    base = getBaseline(row['UserId'], row['MovieId'], trainingRatingMatrix, trainingRatingMatrixT, avgGlobal)
    local = localRating(row['UserId'], row['MovieId'], train, trainingRatingMatrix, trainingRatingMatrixT, avgGlobal, k, simFunc)
    return local + base


# In[ ]:


ParallelRMSE(test, localPlusGlobalPrediction, "localPlusGlobalPrediction", ks, similarityFuncs)


# In[ ]:


#print(getBaseline(363, 97, ratingMatrix, ratingMatrixT, avgGlobal) + 
#      localRatingCutoff(363, 97, ratings_df, ratingMatrix, ratingMatrixT, avgGlobal, .25, pearson))

def localPlusGlobalPredictionCutoff(row, cutoff, simFunc):
    base = getBaseline(row['UserId'], row['MovieId'], trainingRatingMatrix, trainingRatingMatrixT, avgGlobal)
    local = localRatingCutoff(row['UserId'], row['MovieId'], train, trainingRatingMatrix, trainingRatingMatrixT, avgGlobal, cutoff, simFunc)
    return local + base

cutoffs = [.05, .1, .15, .2, .3, .4, .5, .6, .7]

ParallelRMSE(test, localPlusGlobalPredictionCutoff, "localPlusGlobalPredictionCutoff", cutoffs, similarityFuncs)


# In[ ]:


cutoffs = [.7, .8, .9, .95]
similarityFuncs = {'pearson' : pearson}
ParallelRMSE(test, localPlusGlobalPredictionCutoff, "localPlusGlobalPredictionCutoff", cutoffs, similarityFuncs)


# *provide your best 2 or 3 methods, their RMSE, plus some discussion of why they did the best*

# The best method I found was the local plus global function using the jaccard or cosine similarity with a k of 25. This acheived a RMSE of .944 which was .05 better than the base line result (.993). this one probably did better than the rest because it was centered around the global baseline and made incremental improvment where it could. The pearson similarity function seemed to perform badly in this situation.

# ### BONUS: 
# Can you do better? Find a way to improve the results!

# In[ ]:


# your code here

