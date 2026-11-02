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


    # Task 02 addition
    def viterbi(self,x):
        """
        Finds the most likely hidden path and its log2 probability using dynamic programming
        :param x:
        :return:
        """

        if not x:
            return "", -math.inf

        v = []
        back = []

        # Initalising col 0
        v.append({})
        back.append({})
        for k in self.states:
            v[0][k] = self.log_pi[k] + self.log_e[k][x[0]]
            back[0][k] = None

        # Filling the table for positions i>0
        for i in range(1, len(x)):
            v.append({})
            back.append({})
            for k in self.states:
                best_score = -math.inf
                best_prev = None

                # Iteration in self.states order and use > so the
                # first state naturally wins any exact ties
                for prev in self.states:
                    score = v[i-1][prev] + self.log_a[prev][k] + self.log_e[k][x[i]]
                    if score > best_score:
                        best_score = score
                        best_prev = prev

                v[i][k] = best_score
                back[i][k] = best_prev
        # Finding the best ending state in the final col
        best_final_score = -math.inf
        best_final_state = None
        for k in self.states:
            if v[-1][k] > best_final_score:
                best_final_score = v[-1][k]
                best_final_state = k

        # Traceback
        path_list = []
        curr_state = best_final_state
        for i in range(len(x) -1, -1, -1):
            path_list.append(curr_state)
            curr_state = back[i][curr_state]

        # Reverse, as collection was backwards
        path_list.reverse()
        best_path = "".join(path_list)

        return best_path, best_final_score


    # TASK03 addition:
    def forward(self, x):
        """ Computes the Forward table f[i][k] in log2 space, and log2 P(x)"""
        if not x:
            return [], -math.inf

        f = []

        # Initialising column 0
        f.append({})
        for k in self.states:
            f[0][k] = self.log_pi[k] + self.log_e[k][x[0]]

        # Filling the table for i > 0
        for i in range(1, len(x)):
            f.append({})
            for k in self.states:
                log_sum_val = -math.inf
                for prev in self.states:
                    term = f[i-1][prev] + self.log_a[prev][k] + self.log_e[k][x[i]]
                    log_sum_val = log_sum(log_sum_val, term)
                f[i][k] = log_sum_val
        # Total log2 P(x) is the log_sum of the final col
        log2_px = -math.inf
        for k in self.states:
            log2_px = log_sum(log2_px, f[-1][k])

        return f, log2_px


    def backward(self, x):
        """ Computes the backward table b[i][k] in log2 space, and log2 P(x)"""
        if not x:
            return [], -math.inf

        n = len(x)
        b = [{} for _ in range(n)]

        # Initialising last col to log2(1) = 0
        for k in self.states:
            b[n-1][k] = 0.0

        # Filling table backwards
        for i in range(n-2, -1, -1):
            for k in self.states:
                log_sum_val = -math.inf
                for next_state in self.states:
                    term = self.log_a[k][next_state] + self.log_e[next_state][x[i+1]] + b[i+1][next_state]
                    log_sum_val = log_sum(log_sum_val, term)
                b[i][k] = log_sum_val

        # Total log2 P(x) is calculated from col 0
        log2_px = -math.inf
        for k in self.states:
            term = self.log_pi[k] + self.log_e[k][x[0]] + b[0][k]
            log2_px = log_sum(log2_px , term)


        return b, log2_px


    def posterior(self, x):
        """
        Computes the posterior probability P(z_i = k | x) as plain probabilities.
        """
        f, log2_px = self.forward(x)
        b, _ = self.backward(x)

        post = []
        for i in range(len(x)):
            post.append({})
            for k in self.states:
                # 2^(f[i][k] + b[i][k] - log2 P(x))
                log_prob = f[i][k] + b[i][k] - log2_px
                post[i][k] = 2 ** log_prob

        return post

    def posterior_path(self, x):
        """
        Returns the sequence of states with the highest posterior probability at each position.
        """
        post = self.posterior(x)
        path = []

        for i in range(len(x)):
            best_prob = -1.0
            best_state = None
            for k in self.states:
                if post[i][k] > best_prob:
                    best_prob = post[i][k]
                    best_state = k
            path.append(best_state)

        return "".join(path)




# =-=-=-=-=-=-=-= TASK 2 =-=-=-=-=-=-=-=
def read_model(path):
    """ Reads a model file and returns HMM instance """

    with open(path, 'r') as f:
        data = json.load(f)

    return HMM(data["states"], data["alphabet"], data["pi"], data["a"], data["e"])

# =-=-=-=-=-=-=-= TASK 3 =-=-=-=-=-=-=-=
def log_sum(a, b):
    """ Computes log2(2^a + 2^b) safely without leaving log space."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a

    maximum = max(a,b)
    minimum = min(a,b)

    return maximum + math.log2(1+2**(minimum-maximum))

def agreement(z1, z2):
    """
    Returns the fraction of positions where two paths have the same state.
    """
    if not z1 or not z2 or len(z1) != len(z2):
        return 0.0
    matches = sum(1 for a, b in zip(z1, z2) if a == b)
    return matches / len(z1)



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
    try:
        toy = read_model("toy_model.json")
        seq_toy = "phchc"

        print("Test of Viterbi on 'toy_model.json' with seq 'phchc'")
        v_path, v_score = toy.viterbi(seq_toy)

        print(f"\nViterbi path:\t{v_path} (exp. SMMMS)")
        print(f"Viterbi score:\t{v_score:.2f} (exp. -11.87)")

        # Brute force verification
        bf_path, bf_prob, bf_sum = toy.brute_force(seq_toy)
        print(f"\nBrute force path matches ? {v_path == bf_path}")
        print(f"Brute force prob matches? {math.isclose(2**v_score, bf_prob)}")
        print(f"Log joint matches score? {math.isclose(v_score, toy.log_joint(seq_toy, v_path))}")
    except FileNotFoundError:
        print("File 'toy_model.json' not found in the same directory.")

    # =-=-=-=-=-=-=-= TASK 3
    from Bio import SeqIO

    try:
        toy = read_model("toy_model.json")

        # Loading the sequences and paths
        sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse("sequences.fasta", "fasta")}
        true_paths = {rec.id: str(rec.seq) for rec in SeqIO.parse("true_paths.fasta", "fasta")}

        print(f"{'ID':<10} | {'Length':<8} | {'Viterbi Agreement':<18} | {'Posterior Agreement'}")

        # Decoding and comparing every sequence
        for seq_id, seq in sequences.items():
            true_path = true_paths[seq_id]

            # Decoding using both methods
            v_path, _ = toy.viterbi(seq)
            p_path = toy.posterior_path(seq)

            # Calculating agreements
            v_agree = agreement(v_path, true_path)
            p_agree = agreement(p_path, true_path)

            print(f"{seq_id:<10} | {len(seq):<8} | {v_agree:<18.4f} | {p_agree:.4f}")

    except FileNotFoundError as e:
        print(f"File Error: {e}")


"""
THINK ABOUT IT ANSWERS:

Task 2
- when using plain probabilities, values become too small -> Python rounds them down eventually to 0
- this does usually happen for longer sequences 
- when becoming the 0, the algorithm therefore returns arbitrary path because all paths tie at 0 probability 
- the best path then usually costs about 1.5 - 2.5 bits per position 

Task 3
(a)
- The Viterbi optimises global probability of entire continuous path 
meanwhile posterior path optimises local accuracy at each 
individual position 
(b)
- the posterior path almost always has a slightly higher agreement % (~ <0.02%)
- As the posterior decoding explicitly maximises the probability of getting 
each individual position correct, this result is sort of expected of it
(c)
- Viterbi would be better if I would need biologically valid and CONTINUOUS path. As the posterior 
calculates each position independently, it might produce "impossible" results (changing state 
even thought it has 0 probability). Viterbi does guarantee a global and LEGAL path.
"""