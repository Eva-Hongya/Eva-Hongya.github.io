---
layout: post
title:  "Black Friday Sales Analysis and Prediction"
author: Eva
categories: [ Python, Data Visualization, Machine Learning ]
image: assets/images/blackfriday.jpg
featured: true
beforetoc: "Two things I will do in this analysis. First I will draw some insights through exploring the dataset, then I will build a regression model that predicts future purchase."
toc: true
hidden: false
comments: false
---
Spoiler: gender and occupation matter, a lot.

But first thing first, let's see what the data looks like. Below is the first 10 entries of this large dataset (537,577 entries in total). Note: this dataset contains fabricated data provided by Analytics Vidhya. For the full dataset and python code, please visit <a target="_blank" href="https://github.com/Eva-Hongya/Black-Friday-Sales-Analysis-and-Prediction">this github repository</a>.

![walking]({{site.baseurl}}/assets/images/data.png)

#### More men purchase, women purchase more

After a little data wrangling, we have got 5891 unique customers shopped with us during Black Friday sale. Men make up the majority (a bit over 70%), however, women spent a lot (30K) more than their counterpart on average.

![walking]({{site.baseurl}}/assets/images/gender.png) ![walking]({{site.baseurl}}/assets/images/gender2.png)

One thing we want to be careful with is that, sometimes male customers pay the bill when shopping together with women. Unfortunately there is no relevant data that can answer this question. For this analysis let's assume everyone was alone. In this case the data suggests our main target market should be men, also, we should market to women with a different strategy that boosts sales.  

#### Skewed occupation type, similar spending

Our customers have a wide range of occupations. Overall, there is a steep cliff on what job they have. Occupation 4, 0, 7, 1, 17, 12 are the most common ones among our customers,and they exceed other occupations by a great margin.

![walking]({{site.baseurl}}/assets/images/occupation.png)

When it comes to spending, it is a very different story. There is a cluster in how much people of different jobs spend (630K-800K).

People with occupation 18, 19 typically spent a little more than others in our store on Black Friday, although neither of them were in the common occupations for our customers. As a matter of fact, occupation 18 and 19 were next to the least common occupation for our customers. Therefore we should make different marketing strategies for our biggest target (4,0,7,1,17,12) and biggest spender (18,19).

![walking]({{site.baseurl}}/assets/images/occupation2.png)

Other perspectives included in this project:
* Age
* City
* Residence Stability
* Product

For more details, visit <a target = "_blank" href = "https://github.com/Eva-Hongya/Black-Friday-Sales-Analysis-and-Prediction/blob/master/Black%20Friday%20Sales%20Analysis%20and%20Prediction.ipynb">this Jupyter notebook</a>.

#### Purchase Prediction
I used RandomForestRegressor to build the model, but first, I had to find the best parameters, here is the code I used to decide some of the parameters:
```
predictor = data.drop("Purchase", axis=1)
target = data["Purchase"]

# find the best parameter for model making
param_grid = {"n_estimators":[1, 5, 10, 50, 100, 150, 300, 500], \
              "max_depth":[1, 3, 5, 7, 9]}
grid_rf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error").fit(predictor, target)
print("Best parameter: {}".format(grid_rf.best_params_))
print("Best score: {:.2f}".format((-1*grid_rf.best_score_)**0.5))

```
With the parameters I found, I was able to build a model with 96.68% predictability, as shown below.

![walking]({{site.baseurl}}/assets/images/score.png)
