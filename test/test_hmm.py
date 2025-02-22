import pytest
from hmm import HiddenMarkovModel
import numpy as np

def test_mini_weather():
    """
    TODO:
    Create an instance of your HMM class using the "small_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct.

    Ensure that the output of your Viterbi algorithm correct.
    Assert that the state sequence returned is in the right order, has the right number of states, etc.

    In addition, check for at least 2 edge cases using this toy model.
    """

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    observation_sequence = mini_input['observation_state_sequence']
    expected_viterbi_sequence = mini_input['best_hidden_state_sequence']

    hmm_model = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Validate against expected results
    assert hmm_model.forward(observation_sequence) > 0, "Forward probability should be positive"
    assert hmm_model.viterbi(observation_sequence) == list(expected_viterbi_sequence), "Viterbi sequence does not match expected sequence"

    # Edge cases
    assert hmm_model.forward([]) == 0, "Forward probability should be 0 for empty sequence"
    assert hmm_model.viterbi([]) == [], "Viterbi sequence should be empty for empty sequence"

def test_full_weather():

    """
    TODO:
    Create an instance of your HMM class using the "full_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file

    Ensure that the output of your Viterbi algorithm correct.
    Assert that the state sequence returned is in the right order, has the right number of states, etc.

    """
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    hidden_states = full_hmm['hidden_states']
    observation_states = full_hmm['observation_states']
    prior_p = full_hmm['prior_p']
    transition_p = full_hmm['transition_p']
    emission_p = full_hmm['emission_p']

    observation_sequence = full_input['observation_state_sequence']
    expected_viterbi_sequence = full_input['best_hidden_state_sequence']

    hmm_model = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Validate against expected results
    assert hmm_model.forward(observation_sequence) > 0, "Forward probability should be positive"
    assert hmm_model.viterbi(observation_sequence) == list(expected_viterbi_sequence), "Viterbi sequence does not match expected sequence"
    # Edge cases
    same_observation_sequence = [observation_sequence[0]] * len(observation_sequence)
    assert hmm_model.forward(same_observation_sequence) > 0, "Forward probability should be valid for repeated observations"
    assert len(hmm_model.viterbi(same_observation_sequence)) == len(same_observation_sequence), "Viterbi sequence should have correct length"
