import numpy as np
import torch


char_to_id = {
        **{str(i): i for i in range(10)},
        '(': 10,
        ')': 11,
        '+': 12,
        '-': 13,
        '=': 14,
        ' ': 15,
    }

id_to_token = {
        **{i: str(i) for i in range(10)},
        10: "(",
        11: ")",
        12: "+",
        13: "-",
        14: "=",
        15: " ",
    }



def concat_popped_to_previous(A, B, C):
    N, M = A.shape
    A = A.astype(object)  # ensure string operations behave correctly

    A_new = A.copy()

    # Get the elements to be popped
    popped = A[np.arange(N), B]

    # Concatenate to the previous element in each row
    A_new[np.arange(N), B - 1] = "(" + A_new[np.arange(N), B - 1] + C[np.arange(N)] + popped + ")"

    # Mask to remove B[i]-th column from each row
    col_indices = np.arange(M)
    mask = col_indices[None, :] != B[:, None]
    result = A_new[mask].reshape(N, M - 1)

    return result


def concat_op_var(A,C):
    N = A.shape[0]
    A = A.astype(object)  # ensure string operations behave correctly
    A_new = A.copy()
    A_new[np.arange(N), 0] = A_new[np.arange(N), 0] + C[np.arange(N)] + A_new[np.arange(N), 1]
    return A_new[:, :1]

def generate_constrained_sequences(batch_size, T, high=None):
    if high is None:
        high = T
    elif T > high:
        raise ValueError("T cannot be greater than high")

    # Shape: (1, T) → [[high, high-1, ..., high-T+1]]
    upper_bounds = high - np.arange(T)

    # Uniform samples in [0, 1), shape: (batch_size, T)
    random_floats = np.random.rand(batch_size, T)

    # Scale to [1, upper_bound] per timestep
    # floor(random * bound) ∈ [0, bound-1], then +1 ∈ [1, bound]
    samples = np.floor(random_floats * upper_bounds).astype(int) + 1

    return samples


def safe_eval(expr):
    try:
        return str(eval(expr)%10)
    except:
        return expr  # or raise/log if preferred

def transform_strings(arr):
    result = []
    indices = []
    for a in arr:
        s = a[0]
        idx = s.find(")")
        if idx == -1:
            expr = s[0:3]
            r = safe_eval(expr)
            new_s = r + s[3:]
            result.append(new_s)
            indices.append(0)
        else:
            expr = s[idx - 3:idx]
            r = safe_eval(expr)

            if idx >= 4 and s[idx - 4] == "(":
                new_s = s[:idx - 4] + r + s[idx + 1:]
                indices.append(idx - 4)
            else:
                new_s = s[:idx - 3] + r + s[idx:]
                indices.append(idx - 3)
            result.append(new_s)

    return np.array(result), np.array(indices)

def left_pad(arr, Tmax, pad_value=" "):
    B, T = arr.shape
    if T > Tmax:
        raise ValueError("T is already greater than Tmax!")

    # Create full padded array of shape (B, Tmax)
    padded = np.full((B, Tmax), pad_value, dtype=arr.dtype)

    # Copy original values to the right-aligned portion
    padded[:, -T:] = arr
    return padded



def sample_arith_exp(num_samples, num_variables):
    variables = np.random.randint(low=0, high=9, size=(num_samples, num_variables)).astype(str)
    ops = np.random.choice(["+", "-"], size=(num_samples, num_variables-1))
    orders = generate_constrained_sequences(num_samples, num_variables-1)
    if num_variables > 2:
        variables = concat_popped_to_previous(variables, orders[:, 0], ops[:, 0])
    for t in range(1,num_variables-2):
        variables = concat_popped_to_previous(variables, orders[:, t], ops[:, t])
    variables = concat_op_var(variables, ops[:,-1])

    one_step_result, indices = transform_strings(variables.copy())
    
    return variables, one_step_result, indices


def sample_arith_exp_tok(num_samples, num_variables, max_length):
    
    variables, one_step_result, indices = sample_arith_exp(num_samples, num_variables)
    # Vectorized mapping function
    map_func = np.vectorize(lambda c: char_to_id[c])  # default: numeric digit
    
    variables = np.array(list("".join(variables.flatten()))).reshape(variables.shape[0], -1)
    one_step_result = np.array(list("".join(one_step_result.flatten()))).reshape(one_step_result.shape[0], -1)
    variables_mapped = map_func(variables)
    one_step_result_mapped = map_func(one_step_result)
    
    # variables = left_pad(variables, max_length-len(one_step_result[0]))
    variables_mapped = left_pad(variables_mapped, max_length-len(one_step_result[0]), pad_value=15)
    return variables_mapped, one_step_result_mapped, indices


class ArithmeticSampler:
    def __init__(self, max_variables):
        self.max_variables = max_variables
        self.max_length = 8*max_variables - 13
        self.causal_mask = torch.tril(torch.ones((self.max_length, self.max_length))).bool() # (T, T)

    def generate(self, num_samples, get_attn_mask=False):
        probs = np.ones(self.max_variables-1)/(self.max_variables-1)
        draws = np.random.multinomial(n=num_samples, pvals=probs).astype(int)
        
        batch = torch.zeros((num_samples, self.max_length)).long()
        mask = torch.zeros((num_samples, self.max_length)).int() # (B, T+1)
        
        curr_idx = 0
        indices = torch.randperm(num_samples).long()
        
        for i in range(self.max_variables-1):
            if draws[i] == 0: continue
            
            variables_mapped, one_step_result_mapped, ops_indices = sample_arith_exp_tok(draws[i], i+2, self.max_length-1)
            suffix = np.full((draws[i], 1), 14, dtype=variables_mapped.dtype)

            range_vecs = np.arange(1, self.max_length+1)
            # Concatenate along axis 1 (columns)
            variables_mapped = np.concatenate([variables_mapped, suffix], axis=1)
            ops_indices += variables_mapped.shape[1]
            concatenated = np.concatenate((variables_mapped, one_step_result_mapped), axis=1)
            
            batch[indices[curr_idx:curr_idx+draws[i]], :] = torch.from_numpy(concatenated).long()

            mask[indices[curr_idx:curr_idx+draws[i]], :] = torch.from_numpy(range_vecs > len(variables_mapped[0])).int()
            mask[indices[curr_idx:curr_idx+draws[i]], ops_indices] = 2
            curr_idx += draws[i]
        
        if get_attn_mask:
            padding_mask = (batch!=char_to_id[" "]) # (B, T)
            attn_mask = self.causal_mask[None, None, :, :] & padding_mask[:, None, None, :] # (B, 1, T, T)

            return (batch, mask, attn_mask)

        return (batch, mask)
    
    def decode(self, batch):
        batch = batch.cpu().numpy()
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        decoded = ["".join([id_to_token[i] for i in row]) for row in batch]
        return decoded
        