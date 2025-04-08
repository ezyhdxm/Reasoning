import numpy as np
import torch

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

def one_step_reduction(A, B, C):
    N, M = A.shape
    A = A.astype(object)

    A_new = A.copy()

    # Indexing
    idx = np.arange(N)

    # Get elements to combine
    popped = A[idx, B]
    prev = A[idx, B - 1]
    ops = C[idx]

    # Convert to integers
    x = prev.astype(int)
    y = popped.astype(int)

    # Compute result vectorized using np.where
    reduced = np.where(ops == "+", (x + y) % 10, (x - y) % 10).astype(str)

    # Write back to A_new
    A_new[idx, B - 1] = reduced

    # Remove the popped column
    col_indices = np.arange(M)
    mask = col_indices[None, :] != B[:, None]
    result = A_new[mask].reshape(N, M - 1)

    return result

def one_step_reduction_batch_vectorized(A, B, C):
    B_size, T = A.shape
    Tm1 = T - 1

    D = B + np.arange(Tm1)  # shape (B, T-1)
    
    rows = np.arange(B_size)
    merge_step_idx = np.argmin(D, axis=1)  # shape (B,)
    merge_pos = B[rows, merge_step_idx]  # shape (B,)
    ops = C[rows, merge_step_idx]
    left_idx = merge_pos - 1
    right_idx = merge_pos

    # Vectorized gather of left and right
    rows = np.arange(B_size)

    left_tokens = A[rows, left_idx].astype(int)
    right_tokens = A[rows, right_idx].astype(int)
    merged_tokens = np.where(ops == "+", 
                             (left_tokens + right_tokens) % 10,
                             (left_tokens - right_tokens) % 10).astype(str)

    # Build a (B, T) boolean mask to remove the right_idx
    col_indices = np.arange(T)
    mask_keep = col_indices[None, :] != right_idx[:, None]  # (B, T), True means keep

    # Apply mask to reduce A to (B, T-1)
    A_new = A[mask_keep].reshape(B_size, Tm1)
    A_new[rows, left_idx] = merged_tokens  # overwrite the merged position

    # Remove the i-th entry from each B row
    col_indices_b = np.arange(Tm1)
    mask_keep_b = col_indices_b[None, :] != merge_step_idx[:, None]
    col_idx = col_indices_b[None, :]  # shape (1, T-1)
    
    merge_step_idx = merge_step_idx[:, None]    
    mask = col_idx < merge_step_idx
    B[mask] -= 1
    B_new = B[mask_keep_b].reshape(B_size, Tm1 - 1)
    C_new = C[mask_keep_b].reshape(B_size, Tm1 - 1)

    return A_new, B_new, C_new

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

def sample_arith_exp(num_samples, num_variables):
    variables = np.random.randint(low=0, high=9, size=(num_samples, num_variables)).astype(str)
    ops = np.random.choice(["+", "-"], size=(num_samples, num_variables-1))
    orders = generate_constrained_sequences(num_samples, num_variables-1)
    one_step_result, one_step_orders, one_step_ops = one_step_reduction_batch_vectorized(variables.copy(), orders.copy(), ops.copy())
    if num_variables > 2:
        variables = concat_popped_to_previous(variables, orders[:, 0], ops[:, 0])
    for t in range(1,num_variables-2):
        one_step_result = concat_popped_to_previous(one_step_result, one_step_orders[:, t-1], one_step_ops[:, t-1])
        variables = concat_popped_to_previous(variables, orders[:, t], ops[:, t])
    variables = concat_op_var(variables, ops[:,-1])
    if num_variables > 2:
        one_step_result = concat_op_var(one_step_result, one_step_ops[:,-1])
    return variables, one_step_result

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

def left_pad(arr, Tmax, pad_value=" "):
    B, T = arr.shape
    if T > Tmax:
        raise ValueError("T is already greater than Tmax!")

    # Create full padded array of shape (B, Tmax)
    padded = np.full((B, Tmax), pad_value, dtype=arr.dtype)

    # Copy original values to the right-aligned portion
    padded[:, -T:] = arr
    return padded

def sample_arith_exp_tok(num_samples, num_variables, max_length):
    
    variables, one_step_result = sample_arith_exp(num_samples, num_variables)
    # Vectorized mapping function
    map_func = np.vectorize(lambda c: char_to_id[c])  # default: numeric digit
    
    variables = np.array(list("".join(variables.flatten()))).reshape(variables.shape[0], -1)
    one_step_result = np.array(list("".join(one_step_result.flatten()))).reshape(one_step_result.shape[0], -1)
    variables_mapped = map_func(variables)
    one_step_result_mapped = map_func(one_step_result)
    
    variables = left_pad(variables, max_length-len(one_step_result[0]))
    variables_mapped = left_pad(variables_mapped, max_length-len(one_step_result[0]), pad_value=15)
    return variables, one_step_result, variables_mapped, one_step_result_mapped


class ArithmeticSampler:
    def __init__(self, max_variables):
        self.max_variables = max_variables

    def generate(self, num_samples):
        probs = np.ones(self.max_variables-1)/(self.max_variables-1)
        draws = np.random.multinomial(n=num_samples, pvals=probs).astype(int)
        
        MAX_LENGTH = 8*self.max_variables - 14
        batch = torch.zeros((num_samples, MAX_LENGTH+1)).long()
        mask = torch.zeros((num_samples, MAX_LENGTH+1)).bool()
        
        curr_idx = 0
        indices = torch.randperm(num_samples).long()
        
        for i in range(self.max_variables-1):
            if draws[i] == 0: continue
            
            variables, one_step_result, variables_mapped, one_step_result_mapped = sample_arith_exp_tok(draws[i], i+2, MAX_LENGTH)
            suffix = np.full((draws[i], 1), 14, dtype=variables_mapped.dtype)

            range_vecs = np.arange(1, MAX_LENGTH+2)
            # Concatenate along axis 1 (columns)
            variables_mapped = np.concatenate([variables_mapped, suffix], axis=1)
            concatenated = np.concatenate((variables_mapped, one_step_result_mapped), axis=1)
            
            batch[indices[curr_idx:curr_idx+draws[i]], :] = torch.from_numpy(concatenated).long()

            mask[indices[curr_idx:curr_idx+draws[i]], :] = torch.from_numpy(range_vecs > len(variables_mapped[0])).bool()
            curr_idx += draws[i]
        
        return batch, mask
    
    def decode(self, batch):
        batch = batch.cpu().numpy()
        decoded = ["".join([id_to_token[i] for i in row]) for row in batch]
        return decoded
        