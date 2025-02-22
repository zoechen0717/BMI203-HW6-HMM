import numpy as np
from hmm import HiddenMarkovModel

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

def run_hmm_tests(model_file, sequence_file):
    """
    Run forward and Viterbi algorithms using the custom HiddenMarkovModel and print results.
    """
    hidden_states, observation_states, prior_p, transition_p, emission_p = load_hmm_model(model_file)
    observation_sequence, expected_viterbi_sequence = load_test_data(sequence_file)

    hmm_model = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

    # Compute Forward probability
    forward_prob = hmm_model.forward(observation_sequence)
    print(f"Forward Probability: {forward_prob}")

    # Compute Viterbi sequence
    viterbi_sequence = hmm_model.viterbi(observation_sequence)
    print(f"Predicted Viterbi Sequence: {viterbi_sequence}")
    print(f"Expected Viterbi Sequence: {list(expected_viterbi_sequence)}")

    # Check if the predicted sequence matches expected
    if viterbi_sequence == list(expected_viterbi_sequence):
        print("Viterbi sequence matches expected sequence!")
    else:
        print("Viterbi sequence does NOT match expected sequence.")

# Run tests on both datasets
print("Testing Mini Weather HMM")
run_hmm_tests('./data/mini_weather_hmm.npz', './data/mini_weather_sequences.npz')

print("\nTesting Full Weather HMM")
run_hmm_tests('./data/full_weather_hmm.npz', './data/full_weather_sequences.npz')
