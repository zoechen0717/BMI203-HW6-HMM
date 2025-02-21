import numpy as np
from hmmlearn import hmm

def load_hmm_model(file_path):
    """
    Load HMM model parameters from the given .npz file.
    """
    hmm_data = np.load(file_path)
    return (hmm_data['hidden_states'], hmm_data['observation_states'], hmm_data['prior_p'],
            hmm_data['transition_p'], hmm_data['emission_p'])

def load_test_data(file_path):
    """
    Load observation sequence and expected results.
    """
    test_data = np.load(file_path)
    return test_data['observation_state_sequence'], test_data['best_hidden_state_sequence']

def test_hmm_with_hmmlearn(model_file, sequence_file):
    """
    Test HMM using hmmlearn on provided datasets.
    """
    hidden_states, observation_states, prior_p, transition_p, emission_p = load_hmm_model(model_file)
    observation_sequence, expected_viterbi_sequence = load_test_data(sequence_file)

    # Create a mapping of observation states to indices
    observation_states_dict = {state: idx for idx, state in enumerate(observation_states)}

    # Convert observation sequence from strings to one-hot encoded matrix
    numeric_observation_sequence = np.array([observation_states_dict[obs] for obs in observation_sequence])
    one_hot_observation_sequence = np.zeros((len(numeric_observation_sequence), len(observation_states)))
    one_hot_observation_sequence[np.arange(len(numeric_observation_sequence)), numeric_observation_sequence] = 1

    # Initialize hmmlearn HMM model
    sklearn_hmm = hmm.MultinomialHMM(n_components=len(hidden_states), n_trials=1)
    sklearn_hmm.startprob_ = prior_p
    sklearn_hmm.transmat_ = transition_p
    sklearn_hmm.emissionprob_ = emission_p

    # Compute forward probability using hmmlearn
    log_prob = sklearn_hmm.score(one_hot_observation_sequence)
    forward_prob = np.exp(log_prob)
    print(f"Forward Probability: {forward_prob}")

    # Compute Viterbi sequence using hmmlearn
    _, sklearn_viterbi_seq = sklearn_hmm.decode(one_hot_observation_sequence, algorithm="viterbi")
    predicted_sequence = [hidden_states[i] for i in sklearn_viterbi_seq]
    print(f"Predicted Viterbi Sequence: {predicted_sequence}")
    print(f"Expected Viterbi Sequence: {list(expected_viterbi_sequence)}")

    # Validate Viterbi sequence
    assert predicted_sequence == list(expected_viterbi_sequence), "Viterbi sequences do not match"
    print("Test Passed!")

# Run tests for mini and full weather datasets
test_hmm_with_hmmlearn('./data/mini_weather_hmm.npz', './data/mini_weather_sequences.npz')
test_hmm_with_hmmlearn('./data/full_weather_hmm.npz', './data/full_weather_sequences.npz')
