# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming).

![BuildStatus](https://github.com/zoechen0717/BMI203-HW6-HMM/workflows/badge.svg?event=push)

# Assignment

## Overview

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf),
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/).


## Tasks and Data
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class.

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states
* `observation_states`: list of possible observation states
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`)
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences.
 * `observation_state_sequence`: observation sequence to test
* `best_hidden_state_sequence`: correct viterbi hidden state sequence


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions?
  * Are the inputs in compatible shapes/sizes which each other?
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases.

Finally, please update your README with a brief description of your methods.

## Methods description
This project implements a **Hidden Markov Model (HMM)** and provides functionality for running **Forward** and **Viterbi algorithms** to compute probabilities and decode hidden state sequences. The implementation is validated against the `hmmlearn` library for correctness.

### **1. Forward Algorithm**
The **Forward algorithm** calculates the probability of an observed sequence given the model parameters (prior, transition, and emission probabilities). The method follows the standard dynamic programming approach:

#### **Steps:**
  - Initialize the probability matrix `forward[N, T]` where `N` is the number of hidden states and `T` is the sequence length.
  - Compute initial probabilities using prior and emission probabilities.
  - Iteratively update probabilities for each time step using transition and emission probabilities.
  - Compute the final probability by summing over all possible hidden states at the last time step.

#### **Edge Cases Handled:**
- Empty sequences return `0` probability.
- Handles numerical stability using log-space calculations if necessary.

### **2. Viterbi Algorithm**
The **Viterbi algorithm** finds the most likely sequence of hidden states given an observation sequence.

#### **Steps:**
  - Initialize the Viterbi probability table and backpointer table.
  - Compute initial state probabilities using prior and emission probabilities.
  - Iteratively compute the maximum probability for each state at every time step using transition probabilities.
  - Store the best previous state for backtracking.
  - Identify the best final state and reconstruct the optimal hidden state sequence using the backpointer table.

#### **Edge Cases Handled:**
- Empty sequences return an empty state sequence.
- Handles zero transition probabilities correctly.
