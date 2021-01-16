import numpy as np


def shuffle_dataset(x, t):
    # Shuffle dataset. x is the data, t is the label
    p = np.random.permutation(x.shape[0])
    if x.ndim == 2:
        x = x[p, :]
    else:
        x = x[p, :, :, :]

    return x, t[p]


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    # multiple images to 2d array
    # input_data (num, channels, height, width)

    N, C, H, W = input_data.shape

    # Should be an integer
    assert (H + 2 * pad - filter_h) % stride == 0
    assert (W + 2 * pad - filter_w) % stride == 0

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # Do not pad on number of data and channels
    # Pad on the height and width of images
    padding = [(0, 0), (0, 0), (pad, pad), (pad, pad)]
    img = np.pad(input_data, padding, "constant")

    # Create a new matrix. The 0th, 1st arguments N, C are self-explanatory
    # N is responsible for the number of data
    # C is responsible for the number of channels
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # Stretch!
    # Get all values that will be multiplied with (y, x) of the filter
    # There will be filter_h * filter_w values, corresponding to each element of filter
    # which is why the 2nd, 3rd arguments of np.zeros(...) is filter_h, filter_w

    # The values will have size (out_h, out_w), thus the 4th, 5th arguments of
    # np.zeros(...) is out_h, out_w
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # Explanation for (0, 4, 5, 1, 2, 3)
    # Note that this order has been chosen for resizing

    # The resulting 2d-array "should" have dimension
    # (N * out_h * out_w, C * filter_h * filter_w)
    # This 2d-array will be multiplied with the expanded filter,
    # which fill have dimension (C * filter_h * filter_w, fn)
    # (fn: number of filters)
    # Thus the axis order is determined by the shape of 2d-array
    # (N * out_h * out_w, C * filter_h * filter_w)
    # which is 0, 4, 5, 1, 2, 3 (in order of axis represented by each term)
    # 0: Number of data should stay the same
    # 4, 5: out_h * out_w should come first
    # 1: then append the channels
    # 2, 3: filter_h * filter_w

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    # Inverse of im2col
    # Transforms 2d array to 4d input

    N, C, H, W = input_shape

    # Should be an integer
    assert (H + 2 * pad - filter_h) % stride == 0
    assert (W + 2 * pad - filter_w) % stride == 0

    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # Refer to the comments on im2col
    # `col` variable was a 6d-array before reshaping, with dimension
    # (N * out_h * out_w, C * filter_h * filter_w)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)

    # Now we want to transform this back to the original input, which was
    # col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # as written in the implementation of im2col
    # Thus the transpose axis order should be (0, 3, 4, 5, 1, 2)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    # New array to store the image
    # In case of index out of range error, add stride - 1 to
    # 2nd, 3rd dimensions of img
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))

    # Do the exact opposite of im2col
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    # ignore padded values
    return img[:, :, pad : H + pad, pad : W + pad]
