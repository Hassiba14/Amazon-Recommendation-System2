

Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/ 



Let's check the info of the data

There are 7824482 observations and 4 columns in the data.

df.pd.describe()


There are 1540 users in the dataset

There are 5689 items in the dataset
As per the number of unique users and items, there is a possibility of 1540 * 5689 = 8761060 ratings in the dataset. But we only have 65290 ratings, i.e. not every user has rated every item in the dataset. And we can build a recommendation system to recommend items to users which they have not interacted with.


The sum is equal to the total number of observations which implies that there is only interaction between a pair of items and a user.



The Item with ItemId: B0088CJT4U has been interacted by most users which is 206 times.
But still, there is a possibility of 1540-206 = 1334 more interactions as we have 1540 unique users in our datasets. For those 1334 remaining users, we can build a recommendation system to predict who is most likely to interact with the item.
Also, out of these 1334 interactions, we need to consider the distribution of ratings as well.


We can see that this item has been liked by the majority of users, as the count of ratings 5 and 4 is higher than the count of other ratings.
There can be items with very high interactions but the count of ratings 1 and 2 may be much higher than 4 or 5 which would imply that the item is disliked by the majority of users.


The user with userId: ADLVFFE4VBT8 has interacted with the most number of items i.e. 295 times.
But still, there is a possibility of 5689-295 = 5394 more interactions as we have 5689 unique items in our dataset. For those 5394 remaining items, we can build a recommendation system to predict which items are most likely to be watched by this user.


The distribution is higher skewed to the right. Only a few users interacted with more than 50 items.

Now that we have explored and prepared the data, let's start building Recommendation systems.
Question 2: Create Rank-Based Recommendation System (3 Marks)

Model 1: Rank Based Recommendation System

Rank-based recommendation systems provide recommendations based on the most popular items. This kind of recommendation system is useful when we have cold start problems. Cold start refers to the issue when we get a new user into the system and the machine is not able to recommend items to the new user, as the user did not have any historical interactions in the dataset. In those cases, we can use rank-based recommendation system to recommend items to the new user.

To build the rank-based recommendation system, we take average of all the ratings provided to each item and then rank them based on their average rating.

Now, let's create a function to find the top n items for a recommendation based on the average ratings of items. We can also add a threshold for a minimum number of interactions for a item to be considered for recommendation.

We can use this function with different n's and minimum interactions to get items to be recommended.

Recommending top 5 items with 50 minimum interactions based on popularity

We have recommended the top 5 products by using the popularity recommendation system. Now, let's build a recommendation system using collaborative filtering.

Model 2: Collaborative Filtering Based Recommendation System 

In this type of recommendation system, we do not need any information about the users or items. We only need user item interaction data to build a collaborative recommendation system. For example -
1. Ratings provided by users. For example - ratings of books on goodread, movie ratings on imdb etc
2. Likes of users on different facebook posts, likes on youtube videos
3. Use/buying of a product by users. For example - buying different items on e-commerce sites
4. Reading of articles by readers on various blogs

Types of Collaborative Filtering

Similarity/Neighborhood based
User-User Similarity Based
Item-Item similarity based
Model based

Building a baseline user-user similarity based recommendation system

Below, we are building similarity-based recommendation systems using cosine similarity and using KNN to find similar users which are the nearest neighbor to the given user.
We will be using a new library, called surprise, to build the remaining models. Let's first import the necessary classes and functions from this library.

Below we are loading the rating dataset, which is a pandas DataFrame, into a different format called surprise.dataset.DatasetAutoFolds, which is required by this library. To do this, we will be using the classes Reader and Dataset. Finally splitting the data into train and test set.

Now, we are ready to build the first baseline similarity-based recommendation system using the cosine similarity.
KNNBasic is an algorithm that is also associated with the surprise package. It is used to find the desired similar items among a given set of items.


As we can see from above, these baseline model has RMSE=1.05 on test set, we will try to improve this number later by using GridSearchCV tuning different hyperparameters of this algorithm


Let's us now predict rating for an user with userId=0 and for itemId=3906 as shown below

As we can see - the actual rating for this user-item pair is 4 and predicted rating is 4.29 by this similarity based baseline model

Below we are predicting rating for the same userId=0 but for a item which this user has not interacted before i.e. itemId=100, as shown below -

As we can see the estimated rating for this user-item pair is 4.0 based on this similarity based baseline model

Improving similarity based recommendation system by tuning its hyper-parameters

Below we will be tuning hyper-parmeters for the KNNBasic algorithms. Let's try to understand different hyperparameters of KNNBasic algorithm -

k (int) – The (max) number of neighbors to take into account for aggregation (see this note). Default is 40.
min_k (int) – The minimum number of neighbors to take into account for aggregation. If there are not enough neighbors, the prediction is set to the global mean of all ratings. Default is 1.
sim_options (dict) – A dictionary of options for the similarity measure. And there are four similarity measures available in surprise -
cosine
msd (default)
pearson
pearson baseline
For more details please refer the official documentation https://surprise.readthedocs.io/en/stable/knn_inspired.html



Once the grid search is complete, we can get the optimal values for each of those hyperparameters as shown above.

Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters

Now, let's build the final model by using tuned values of the hyperparameters, which we received by using grid search cross-validation.


We can see from above that after tuning hyperparameters, RMSE for testset has reduced to 0.98 from 1.05. We can say that we have been able to improve the model after hyperparameter tuning



Let's us now predict rating for an user with userId=0 and for itemId=3906 with the optimized model as shown below


If we compare the above predicted rating, we can see the baseline model predicted rating as 4.29 and the optimized model predicted the rating as 4.29.

Below we are predicting rating for the same userId=0 but for a item which this user has not interacted before i.e. itemId=100, by using the optimized model as shown below -


If we compare the above predicted rating, we can see the baseline model predicted rating as 4 and the optimized model predicted the rating as 4.29.

Identifying similar users to a given user (nearest neighbors)

We can also find out the similar users to a given user or its nearest neighbors based on this KNNBasic algorithm. Below we are finding 5 most similar user to the userId=0 based on the msd distance metric

Implementing the recommendation algorithm based on optimized KNNBasic model

Below we will be implementing a function where the input parameters are -
data: a rating dataset
user_id: an user id against which we want the recommendations
top_n: the number of items we want to recommend
algo: the algorithm we want to use to predict the ratings

Predicted top 5 items for userId=4 with similarity based recommendation system


Model 3: Item based Collaborative Filtering Recommendation System 


As we can see from above, these baseline model has RMSE=1.06 on test set, we will try to improve this number later by using GridSearchCV tuning different hyperparameters of this algorithm
Let's us now predict rating for an user with userId=0 and for itemId=3906 and itemId=100

As we can see - the actual rating for this user-item pair is 4 and predicted rating is 4.29 by this similarity based baseline model

Let's predict the rating for the same userId=0 but for a item which this user has not interacted before i.e. itemId=22607

As we can see the estimated rating for this user-item pair is 5.00 based on this similarity based baseline model


Once the grid search is complete, we can get the optimal values for each of those hyperparameters as shown above

Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters

Now let's build the final model by using tuned values of the hyperparameters which we received by using grid search cross-validation.


We can see from above that after tuning hyperparameters, RMSE for testset has reduced to 0.98 from 1.06. We can say that we have been able to improve the model after hyperparameter tuning.



Let's us now predict rating for an user with userId=0 and for itemId=3906 with the optimized model as shown below


If we compare the above predicted rating, we can see the baseline model predicted rating as 4.29 and the optimized model predicted the rating as 4.29. whereas the actual rating is 4.0.

Let's predict the rating for the same userId=0 but for a item which this user has not interacted before i.e. itemId=100, by using the optimized model:


If we compare the above predicted rating, we can see the baseline model predicted rating as 5 and the optimized model predicted the rating as 4.29.

Identifying similar items to a given item (nearest neighbors)
We can also find out the similar items to a given item or its nearest neighbors based on this KNNBasic algorithm. Below we are finding 5 most similar item to the itemId=100 based on the msd distance metric

Predicted top 5 items for userId=4 with similarity based recommendation system



Model 4: Based Collaborative Filtering - Matrix Factorization using SVD 

Model-based Collaborative Filtering is a personalized recommendation system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information. We use latent features to find recommendations for each user.

Latent Features: The features that are not present in the empirical data but can be inferred from the data. For example:

Singular Value Decomposition (SVD)

SVD is used to compute the latent features from the user-item matrix. But SVD does not work when we miss values in the user-item matrix.

Building a baseline matrix factorization recommendation system


We can that the baseline RMSE for matrix factorization model on testset (which is 0.93) is lower as compared to the RMSE for baseline similarity based recommendation system (which is 1.06) and it is even lesser than the RMSE for optimized similarity based recommendation system (which is 0.99)

Let's us now predict rating for an user with userId=0 and for itemId=3906 as shown below

As we can see - the actual rating for this user-item pair is 4 and predicted rating is 4.71 by this matrix factorization based baseline model. It seems like we have over estimated the rating by a small margin. We will try to fix this later by tuning the hyperparameters of the model using GridSearchCV

Below we are predicting rating for the same userId=0 but for a item which this user has not interacted before i.e. userId=100, as shown below -


We can see that estimated rating for this user-item pair is 4.65 based on this matrix factorization based baseline model.

Improving matrix factorization based recommendation system by tuning its hyper-parameters

In SVD, rating is predicted as -

r^ui​=μ+bu​+bi​+qiT​pu​

If user u is unknown, then the bias bu​ and the factors pu​ are assumed to be zero. The same applies for item i with bi​ and qi​.

To estimate all the unknown, we minimize the following regularized squared error:

rui​∈Rtrain ​∑​(rui​−r^ui​)2+λ(bi2​+bu2​+∥qi​∥2+∥pu​∥2)

The minimization is performed by a very straightforward stochastic gradient descent:

bu​bi​pu​qi​​←bu​+γ(eui​−λbu​)←bi​+γ(eui​−λbi​)←pu​+γ(eui​⋅qi​−λpu​)←qi​+γ(eui​⋅pu​−λqi​)​

There are many hyperparameters to tune in this algorithm, you can find a full list of hyperparameters here

Below we will be tuning only three hyperparameters -
n_epochs: The number of iteration of the SGD algorithm
lr_all: The learning rate for all parameters
reg_all: The regularization term for all parameters


Once the grid search is complete, we can get the optimal values for each of those hyperparameters, as shown above.

Below we are analysing evaluation metrics - RMSE and MAE at each and every split to analyze the impact of each value of hyperparameters

Now, we will the build final model by using tuned values of the hyperparameters, which we received using grid search cross-validation above.



Let's us now predict rating for an user with userId=0 and for itemId=3906 with the optimized model as shown below


If we compare the above predicted rating, we can see the baseline model predicted rating as 4.71 and the optimized model predicted the rating as 4.86. whereas the actual rating is 4.



Predicting ratings for already interacted items

Below we are comparing the rating predictions of users for those items which has been already watched by an user. This will help us to understand how well are predictions are as compared to the actual ratings provided by users

Here we are comparing the predicted ratings by similarity based recommendation system against actual ratings for userId=4

We can see that distribution of predicted ratings is closely following the distribution of actual ratings. The total bins for predicted ratings are nearly same as to the total bins for actual ratings.
We are getting more predicted values in between 4 and 5,this is expected, as actual ratings always have discreet values like 1, 2, 3, 4, 5, but predicted ratings can have continuous values as we are taking aggregated ratings from the nearest neighbors of a given user. But over the predictions looks good as compared to the distribution of actual ratings.

Below we are comparing the predicted ratings by matrix factorization based recommendation system against actual ratings for userId=4

Precision and Recall @ k
RMSE is not the only metric we can use here. We can also examine two fundamental measures, precision and recall. We also add a parameter k which is helpful in understanding problems with multiple rating outputs.
Precision@k - It is the fraction of recommended items that are relevant in top k predictions. Value of k is the number of recommendations to be provided to the user. One can choose a variable number of recommendations to be given to a unique user.
Recall@k - It is the fraction of relevant items that are recommended to the user in top k predictions.
Recall - It is the fraction of actually relevant items that are recommended to the user i.e. if out of 10 relevant movies, 6 are recommended to the user then recall is 0.60. Higher the value of recall better is the model. It is one of the metrics to do the performance assessment of classification models.
Precision - It is the fraction of recommended items that are relevant actually i.e. if out of 10 recommended items, 6 are found relevant by the user then precision is 0.60. The higher the value of precision better is the model. It is one of the metrics to do the performance assessment of classification models.
See the Precision and Recall @ k section of your notebook and follow the instructions to compute various precision/recall values at various values of k.
To know more about precision recall in Recommendation systems refer to these links :
https://surprise.readthedocs.io/en/stable/FAQ.html
https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54

Collaborative Filtering using user-user based interaction performed well in both the k values with Precision value ~84% (k=10) and with k=5, ~86%.
Tuned SVD has better RMSE than all models but Collaborative Filtering using user-user based interaction is also giving good results based on Precsion and recall @k for K=10.
The final model will denpend on the business requirements as whether they have to minimize RMSE or go with maximizing Precision/Recall.


Compare the results from the base line user-user and item-item based models.

 The matrix factorization model is different from the collaborative filtering models. Briefly describe this difference. Also, compare the RMSE and precision recall for the models.
 Does it improve? Can you offer any reasoning as to why that might be?

User-based and Item-based Collaborative Models have nearly same. User based RMSE values (1.05) while the "Item based" model's RMSE is 1.06. Clearly, tuned Collaborative Filtering Models have performed better than baseline model and the user-user based tuned model is performing better and have rmse of 0.9887
The Collaborative Models use the user-item-ratings data to find similarities and make predictions rather than just predicting a random rating based on the distribution of the data. This could a reason why the Collaborative filtering performed well.
Collaborative Filtering searches for neighbors based on similarity of item (example) preferences and recommend items that those neighbors interacted while Matrix factorization works by decomposing the user-item matrix into the product of two lower dimensionality rectangular matrices.
RMSE for Matrix Factorization (0.92) is better than the Collaborative Filtering Models (~1.00).
Tuning SVD matrix factorization model is not improving the base line SVD much.
Matrix Factorization has lower RMSE due to the reason that it assumes that both items and users are present in some low dimensional space describing their properties and recommend a item based on its proximity to the user in the latent space. Implying it accounts for latent factors as well.

Conclusions

In this case study, we saw three different ways of building recommendation systems:
rank-based using averages
similarity-based collaborative filtering3+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-++++++++++++-
model-based (matrix factorization) collaborative filtering

We also understood advantages/disadvantages of these recommendation systems and when to use which kind of recommendation systems. Once we build these recommendation systems, we can use A/B Testing to measure the effectiveness of these systems.

Here is an article explaining how Amazon use A/B Testing to measure effectiveness of its recommendation systems.


<class 'pandas.core.frame.DataFrame'> RangeIndex: 7824482 entries, 0 to 7824481 Data columns (total 3 columns): # Column Dtype --- ------ ----- 0 user_id object 1 item_id object 2 rating float64 dtypes: float64(1), object(2) memory usage: 179.1+ MB 

