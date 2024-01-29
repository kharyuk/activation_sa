import math
import copy
import numpy as np

def perm_recursion(ind, sequence, result):
    N = len(sequence)
    if N == 0:
        return []
    #sequence.sort()
    cur_fact = math.factorial(N-1)
    new_ind, rem = ind // cur_fact, ind % cur_fact
    result.append(sequence[new_ind])
    del sequence[new_ind];
    if rem == 0:
        return result+sequence
    return perm_recursion(rem, sequence, result)

def permute_by_index(ind, sequence):
    N = len(sequence)
    assert 0 <= ind < math.factorial(N)
    return perm_recursion(ind, copy.copy(sequence), [])

class RandomPermutedCompose:
    '''
    Based on torchvision.transforms.Compose. The difference
    is in considering the randm permutation of the given
    transformations.
    '''

    def __init__(self, transforms, permutation_seed=0):
        self.transforms = transforms
        self.N = len(self.transforms)
        self.seed(permutation_seed)
        
    def seed(self, random_seed):
        self.rng = np.random.default_rng(random_seed)
        
    def sample_permutation(self):
        ind = self.rng.choice(self.N, shuffle=True)
        sigma = np.arange(self.N).tolist()
        return permute_by_index(ind, sigma)

    def __call__(self, img):
        sigma = self.sample_permutation()
        for i in sigma:
            img = self.transforms[i](img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string