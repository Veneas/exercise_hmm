import math
import json
import itertools


# =-=-=-=-=-=-=-= TASK 1 =-=-=-=-=-=-=-=
def log2(p):
    """Returning the math.log2(p), or -math.inf when p is 0"""
    if p == 0:
        return -math.inf
    return math.log2(p)


class HMM:
    def __init__(self, states, alphabet,
                 pi, a, e):
        """Initialize the HMM object and precomputes log2 prob values"""
        self.states = states
        self.alphabet = alphabet
        self.pi = pi
        self.a = a
        self.e = e

        # Self-checks: Dicts have to perfectly match the states/alphabet
        assert math.isclose(sum(self.pi.values()), 1.0)
        assert set(self.pi.keys()) == set(self.states)

        for state in self.states:
            assert set(self.a[state].keys()) == set(self.states)
            assert math.isclose(sum(self.a[state].values()), 1.0)

            assert set(self.e[state].keys()) == set(self.alphabet)
            assert math.isclose(sum(self.e[state].values()), 1.0)

        # Precomputing the log2 version of tables
        self.log_pi = {s: log2(p) for s, p in self.pi.items()}
        self.log_a = {s1: {s2: log2(p) for s2, p in self.a[s1].items()} for s1 in self.states}
        self.log_e = {s: {char: log2(p) for char, p in self.e[s].items()} for s in self.states}


    def joint_probability(self,x, z):
        """
        Calculates P(x, z) using floating-point multiplication.
        :param x: Sequence (string)
        :param z: hidden path (string)
        :return:
        """

        if not x or not z or len(x) != len(z):
            return 0.0

        prob = self.pi[z[0]] * self.e[z[0]][x[0]]
        for i in range(1, len(x)):
            prob *= self.a[z[i-1]][z[i]] * self.e[z[i]][x[i]]


        return prob

    def log_joint(self, x, z):
        """
        Calculating log2 P(X, z) by summing precomputed log vals
        """

        if not x or not z or len(x) != len(z):
            return -math.inf

        score = self.log_pi[z[0]] + self.log_e[z[0]][x[0]]
        for i in range(1, len(x)):
            score += self.log_a[z[i-1]][z[i]] + self.log_e[z[i]][x[i]]

        return score


    def brute_force(self, x):
        """
        Checking every possible path to find the most likely one,
        its probability, and the sum of all path probs.
        :param x:
        :return:
        """

        best_path = ""
        best_prob = -1.0
        sum_prob = 0.0

        # itertools.product generates every possible comb of states
        for path_tuple in itertools.product(self.states, repeat=len(x)):
            path = "".join(path_tuple)
            p = self.joint_probability(x, path)

            sum_prob += p
            if p > best_prob:
                best_prob = p
                best_path = path

        return best_path, best_prob, sum_prob


def read_model(path):
    """ Reads a model file and returns HMM instance """

    with open(path, 'r') as f:
        data = json.load(f)

    return HMM(data["states"], data["alphabet"], data["pi"], data["a"], data["e"])






# =-=-=-=-=-=-=-= TASK 2 =-=-=-=-=-=-=-=





# =-=-=-=-=-=-=-= TASK 3 =-=-=-=-=-=-=-=







# =-=-=-=-=-=-=-= MAIN =-=-=-=-=-=-=-=
if __name__ == "__main__":
    # =-=-=-=-=-=-=-= TASK 1
    # Loading the weather model
    try:
        weather = read_model("weather_model.json")
        print(f"Weather model loaded."
              f"States: {weather.states},"
              f"Alphabet: {weather.alphabet}")

        # Checking joint probabilities
        seq = "sr"
        path = "HL"
        jp = weather.joint_probability(seq, path)
        lj = weather.log_joint(seq, path)
        print(f"\nSequence: {seq}, | Path: {path}")
        print(f"Joint probability:\t{jp}")
        print(f"Log joint:\t{lj:.2f} (exp. ~ 4.21)")
        print(f"Math check:\t{math.isclose(lj, log2(jp))}")

        # Checking brute force
        long_seq = "ssrssrrrss"
        best_path, best_prob, sum_prob = weather.brute_force(long_seq)
        print(f"\nBrute force on '{long_seq}':")
        print(f"Best path:\t{best_path} (exp. HHHHHLLLHH)")
        print(f"Best probability:\t{best_prob:.2e} (exp. 5.53e-05)")
        print(f"Sum prob:\t{sum_prob:.6f} (exp. 0.000822)")

    except FileNotFoundError:
        print("File 'weather_model.json' not found in the same directory.")
    # =-=-=-=-=-=-=-= TASK 2
    # =-=-=-=-=-=-=-= TASK 3