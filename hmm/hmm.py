import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states
            hidden_states (np.ndarray): hidden states
            prior_p (np.ndarray): prior probabities of hidden states
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states
        """

        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}

        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence
        """
        if len(input_observation_states) == 0:
        return 0  # Return 0 probability for empty sequence
        
        # Inspired by the psedocode from the pdf file: https://web.stanford.edu/~jurafsky/slp3/A.pdf
        # Get dimentions for probability table
        T = len(input_observation_states)
        N = len(self.hidden_states)

        # Step 1. Initialize variables
        # Initialize probability table
        forward = np.zeros((N, T))
        for s in range(N):
            # Update probabilities based on previous states
            forward[s, 0] = self.prior_p[s] * self.emission_p[s, self.observation_states_dict[input_observation_states[0]]]

        # Step 2. Calculate probabilities
        for t in range(1, T):  # Iterate over time steps
            for s in range(N):  # Iterate over hidden states
                forward[s, t] = np.sum(forward[:, t - 1] * self.transition_p[:, s]) * self.emission_p[s, self.observation_states_dict[input_observation_states[t]]]
        # Step 3. Return final probability
        # Sum up all probabilities and return forward probability
        forward_probability = np.sum(forward[:, -1])
        return forward_probability


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        # Step 1. Initialize variables
        N = len(self.hidden_states)
        T = len(decode_observation_states)
        # store probabilities of hidden state at each step
        viterbi_table = np.zeros((N, T))
        # store best path for traceback
        backpointer = np.zeros((N, T), dtype=int)
        # compute initial probabilities
        for s in range(N):
            viterbi_table[s, 0] = self.prior_p[s] * self.emission_p[s, self.observation_states_dict[decode_observation_states[0]]]
            backpointer[s, 0] = 0  # No previous state at time 0

        # Step 2. Calculate Probabilities
        # update probabilities and store best paths
        for t in range(1, T):
            for s in range(N):
                # Compute max probability for reaching state `s` at time `t`
                prob_values = viterbi_table[:, t-1] * self.transition_p[:, s]
                best_prev_state = np.argmax(prob_values)

                # Update viterbi table and backpointer
                viterbi_table[s, t] = prob_values[best_prev_state] * self.emission_p[s, self.observation_states_dict[decode_observation_states[t]]]
                backpointer[s, t] = best_prev_state

        # find the best final state with the highest probability
        best_last_state = np.argmax(viterbi_table[:, -1])
        bestpathprob = np.max(viterbi_table[:, -1])

        # Step 3. Traceback
        # best hidden state sequence using the stored best path
        best_hidden_state_sequence = [best_last_state]
        for t in range(T - 1, 0, -1):
            best_hidden_state_sequence.insert(0, backpointer[best_hidden_state_sequence[0], t])

        # Step 4. Return best hidden state sequence
        # make index sequence to actual hidden state labels
        best_hidden_state_sequence = [self.hidden_states[int(s)] for s in best_hidden_state_sequence]

        return best_hidden_state_sequence
