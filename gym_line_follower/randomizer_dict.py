import random


class RandomizerDict(dict):
    """
    Dict with randomizing function. If the value of a key is a list/tuple the value is randomized based on length
    of the list/tuple:
        [] --> [] does nothing
        [x] --> x remove container
        [x, y] --> a = random uniform sample on interval [x, y]
        [x, y, ..., z] --> b = random choice ofe one value from the list
    Retains all other dict functionality,
    """
    def __init__(self, *args, **kwargs):
        super(RandomizerDict, self).__init__(*args, **kwargs)
        self.original = self.copy()
        self.randomize()

    def randomize(self, seed=None):
        """
        Randomize dict values.
        :param seed: random generator seed
        :return: None
        """
        random.seed(seed)
        for key, value in self.original.items():
            if isinstance(value, (list, tuple)):
                if len(value) < 1:
                    continue
                elif len(value) == 1:
                    self[key] = value[0]
                elif len(value) == 2:
                    self[key] = random.uniform(value[0], value[1])
                else:
                    self[key] = random.choice(value)


if __name__ == '__main__':
    data = {"a": 1.232,
            "b": [2., 3.],
            "c": [10, 20, 30, 40, 50],
            "d": [42.23]}
    config = RandomizerDict(data)
    print(data)
    for i in range(10):
        config.randomize()
        print(config)
