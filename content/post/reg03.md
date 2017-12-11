---
title: "Feature reconstruction in the Porto Seguro Kaggle competition"
date: 2017-12-07T20:54:06+01:00
draft: false
coverImage: /images/cover3.jpg
coverSize: partial
thumbnailImage: images/seguro.jpg
thumbnailImagePosition: left
---

The [Porto Seguro](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction) Kaggle competition featured data, which was anonymized to the point, where the actual meaning of the features was obscured. So well in fact, that during the competition only very few features were explained, making manual feature engineering essentially impossible. As a result, the winning solutions were based on neural networks, which learned good representations and avoided feature engineering entirely.

I managed to reverse engineer the transformation, which had been applied to many features of the data set and posted my findings in a Kaggle hosted kernel [here](https://www.kaggle.com/pnagel/reconstruction-of-ps-reg-03).
