---
title: 'CSE 5523: Machine Learning'
author: "Yuvraj Singh"
date: "12/10/2021"
output:
  pdf_document: default
  html_document: default
geometry: margin=2.54cm
header-includes: \usepackage{setspace}\doublespacing
---

<style type="text/css">
  body{
  font-size: 11pt;
}
</style>

# CSE 5523: Machine Learning and Statistical Pattern Recognition

Final Project: Road Accident Severity Classification Using US Accidents Dataset

Yuvraj Singh (singh.1250) and Abdirashid Dahir (dahir.39)

## Abstract

Fewer drivers were on the roads during the onset of COVID-19 pandemic because of the stay-at-home mandates and workplace policies aimed at curbing the viral spread. These stay-at-home policies led to a significant reduction in car traffic, hence reducing urban traffic congestion. This study aims to classify the severity of road accidents using the US Accidents dataset [link here](https://www.kaggle.com/sobhanmoosavi/us-accidents) covering road accidents from 49 contiguous US states from February 2016 through December 2020. The severity of road accidents can be qualitatively described to come from one of these classes: low, medium and high severity. This is an exploratory study employing several machine learning approaches including ensemble methods: random forests, heuristic multi-class classifiers: One vs All and One vs One strategy for perceptron and support vector machine binary classifiers and multi-layer neural networks. Model evaluation is based on confusion matrices and the models are validated using cross-validation methods and binomial significance testing.

- [CSE 5523: Machine Learning and Statistical Pattern Recognition](#cse-5523-machine-learning-and-statistical-pattern-recognition)
  - [Abstract](#abstract)
  - [Introduction: Dataset Description](#introduction-dataset-description)
  - [Feature Selection](#feature-selection)
  - [Base Model: Logistic Regression](#base-model-logistic-regression)
  - [Methodologies: Model Evaluation](#methodologies-model-evaluation)
  - [Models](#models)
    - [Random Forest with Bootstrap Aggregation](#random-forest-with-bootstrap-aggregation)
    - [Random Forest with Adaptive Boosting](#random-forest-with-adaptive-boosting)
    - [Multi-Class Classification using Heuristic Approaches](#multi-class-classification-using-heuristic-approaches)
      - [Perceptron: One vs All and One vs One](#perceptron-one-vs-all-and-one-vs-one)
      - [Support Vector Machine: One vs All and One vs One](#support-vector-machine-one-vs-all-and-one-vs-one)
    - [Fully-Connected Multi-Layer Neural Networks](#fully-connected-multi-layer-neural-networks)
  - [Model Performance](#model-performance)
    - [Cross Validation](#cross-validation)
    - [Binomial Significance Testing](#binomial-significance-testing)
  - [Discussion](#discussion)
  - [Division of Work](#division-of-work)
  - [References](#references)

## Introduction: Dataset Description

Dataset URL: [https://www.kaggle.com/sobhanmoosavi/us-accidents][https://www.kaggle.com/sobhanmoosavi/us-accidents]

## Feature Selection

## Base Model: Logistic Regression

## Methodologies: Model Evaluation

## Models

### Random Forest with Bootstrap Aggregation

### Random Forest with Adaptive Boosting

### Multi-Class Classification using Heuristic Approaches

#### Perceptron: One vs All and One vs One

#### Support Vector Machine: One vs All and One vs One

### Fully-Connected Multi-Layer Neural Networks

## Model Performance

### Cross Validation

### Binomial Significance Testing

## Discussion

## Division of Work

1. Yuvraj
   1. Feature Selection: data cleanup
   2. Base Model: Logistic Regression
   3. Random Forest with Adaptive Boosting
   4. Perceptron: One vs. All, One vs. One
   5. Neural Networks
   6. Confusion matrix metrics and cross-Validation
   7. Integrating the main script to run all models and validation
2. Abdirashid
   1. Feature Selection: correlation matrix and selection criteria
   2. Random Forest with Bootstrap Aggregation
   3. Support Vector Machine: One vs. All, One vs. One
   4. Binomial significance testing
   5. Credible intervals of test accuracy

## References
