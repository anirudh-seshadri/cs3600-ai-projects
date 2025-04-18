# Assignment-3

Assignment 3: Inference and Language Modeling for CS 3600 (Intro to AI) in the Spring 2025 semester!

This repository contains all the files you'll need to get started on Assignment 3. This assignment will have you implement a Bayesian Network using `pgmpy`, complete inference, and perform some basic language modeling and text generation using N-grams. This assignment is split into two parts. Part A is worth 50 points, and Part B is worth 50 points. The overall assignment, combining the two parts for a total of 100 points, will be worth 12.5% of your final grade for Assignment 3.

> You are not permitted to fork this repository, as it will create a public repository that you cannot change the visibility of. If you wish to store your code in a GitHub repository, download the source code from this repository, and upload it to a new, private repository on GitHub Enterprise or your personal GitHub account.

## Part A - Inference

In Part A of Assignment 3, you will first explore a lab safety scenario. You are given a scenario description of key events (variables) and their relationships in [the following file](A3-A/scenario.md). You will then implement a Bayesian Network using `pgmpy` to model the relationships between all the events. You will then perform inference on the Bayesian Network to answer a series of questions about the scenario.

Next, you will explore a ghost hunting scenario in the TextWorld environment. The environment features a hidden ghost that might be inside a room or inside the walls (not in a room), but you can hear the ghost make noise. Whenever an action is taken, your agent will receive information about how far the ghost sounds, but this perceived distance is noisy. Luckily, you're also given a special sensor that gives you a distribution of the horizontal distance of the ghost from the origin of the grid. Your goal is to design an agent that can confidently locate where the ghost is. 

This portion of the assignment is worth 50 points, out of 100 total points for the entire assignment. Your code for the lab safety scenario and the ghost hunting scenario will be tested separately. The lab safety scenario is worth 15 points, and there are no hidden tests for this part, as the probabilities are provided in the scenario description. What you see in Gradescope upon submission is your score for that scenario. The ghost hunting scenario is worth 35 points. There are hidden tests for this part of the assignment (i.e., we will be testing your agent on other random seeds and environments, but all the parameters will be similar to the ones provided). You must test your code on further environments to ensure it works correctly. We have provided you with some sample environments to test your agent on - performance on these environments is good indication, but not a guarantee of any final score. 

## Part B - Language Modeling

In Part B of Assignment 3, you will be implementing a simple language model using N-grams. You will first implement a unigram model, then a bigram model, and finally a trigram model. You will then use these models to generate text. You will also implement a function to calculate the perplexity of a given text using your language model. 

This portion of the assignment is worth 50 points, out of 100 total points for the entire assignment. Your code will be tested on larger samples from the same texts provided, and there will be several trials for each model. We have provided you with some sample text files for testing your models. You should test your models on these files, as well as other text files you find online. If you write further tests on other text files, you are permitted to share the results of these tests with your peers. As a reminder, you are not permitted to share your code - all code written must be your own. 

## Misc.

If you haven't already, make sure you get your development environment setup by following the instructions [here](https://github.gatech.edu/CS-3600-Spring-25/dev-environment-setup).

For any questions, please post on Ed Discussion or come to office hours. We highly recommend making public posts on Ed Discussion to promote discussion between students. Furthermore, public posts make you communicate your questions without simply pasting your code, which is a great way to learn and improve your problem-solving & technical communication skills.