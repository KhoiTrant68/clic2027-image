"""Exact uint8 block-moment projection with gamut constraints; a known-method control."""
import numpy as np


def blocks(image, block):
    h, w, c = image.shape
    if c != 3 or h % block or w % block:
        raise ValueError('RGB dimensions must be divisible by block size')
    return image.reshape(h//block, block, w//block, block, c).transpose(0, 2, 4, 1, 3).reshape(-1, block*block)


def unblocks(values, shape, block):
    h, w, c = shape
    return values.reshape(h//block, w//block, c, block, block).transpose(0, 3, 1, 4, 2).reshape(shape)


def project(proposal, reference, block=16, random_moment=False):
    if proposal.dtype != np.uint8 or reference.dtype != np.uint8 or proposal.shape != reference.shape:
        raise ValueError('Require equal-shape uint8 images')
    candidate = blocks(proposal, block).astype('float64')
    anchor = blocks(reference, block).astype('int64')
    signs = np.ones((1, block*block), dtype='int64')
    if random_moment:
        signs = np.random.default_rng(2037+block).choice([-1, 1], size=signs.shape)
    # Flipping negative-sign coordinates converts signed-moment conservation
    # into ordinary sum conservation, preserving the [0,255] box.
    candidate = np.where(signs > 0, candidate, 255-candidate)
    anchor = np.where(signs > 0, anchor, 255-anchor)
    target = anchor.sum(1)
    lo, hi = np.full((len(candidate), 1), -255.), np.full((len(candidate), 1), 255.)
    for _ in range(40):
        mid = (lo+hi)/2
        below = np.clip(candidate+mid, 0, 255).sum(1, keepdims=True) < target[:, None]
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    real = np.clip(candidate+(lo+hi)/2, 0, 255)
    integer = np.floor(real).astype('int64')
    k = target-integer.sum(1)
    if (k < 0).any() or (k > block*block).any():
        raise ValueError('Infeasible balanced rounding')
    order = np.argsort(-(real-integer), axis=1, kind='stable')
    increments = np.zeros_like(integer)
    np.put_along_axis(increments, order, (np.arange(block*block)[None, :] < k[:, None]).astype('int64'), axis=1)
    result = integer+increments
    if not np.array_equal(result.sum(1), target) or result.min() < 0 or result.max() > 255:
        raise AssertionError('Projection/rounding violated exact moment or gamut')
    result = np.where(signs > 0, result, 255-result).astype('uint8')
    image = unblocks(result, reference.shape, block)
    if not np.array_equal((blocks(image, block)*signs).sum(1), (blocks(reference, block)*signs).sum(1)):
        raise AssertionError('Final uint8 invariant failed')
    return image


def self_check():
    rng = np.random.default_rng(2037)
    a = rng.integers(0, 256, size=(32, 32, 3), dtype='uint8')
    b = rng.integers(0, 256, size=a.shape, dtype='uint8')
    for size in [4, 8, 16]:
        for random in [False, True]:
            for reference in [a, np.zeros_like(a), np.full_like(a, 255)]:
                result = project(b, reference, size, random)
                assert result.shape == a.shape and result.dtype == np.uint8
    assert np.array_equal(project(a, a), a)
    print('exact uint8 moments, signed equal-rank control, gamut extremes and identity passed')


if __name__ == '__main__':
    self_check()
