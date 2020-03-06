import numpy as np

def iterate_minibatches(inputs, minisize, shuffle=False):
    fullsize = inputs[0].shape[0]
    assert all([fullsize == other.shape[0] for other in inputs[1:]])
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, fullsize, minisize):
        end_idx = min(start_idx + minisize, fullsize)
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield [i[excerpt] for i in inputs]