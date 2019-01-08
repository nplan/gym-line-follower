import random


class RandomizerDict(dict):
    """
    Dict with randomizing function. Supports int, float, str, bool.
    If value randomization at a key is desired, provided value must be a dict with entries:
        'range': [a, b] - random uniform sample in range a, b
        'choice': [a, b, c, d, ...] - random choice of one value from list
        'default': a - default value
    Default value must always be provided. One of keys 'range' or 'choice' must be provided.
    If value at key is not a dict no randomization is performed.
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
        """
        random.seed(seed)
        for key, value in self.original.items():
            if isinstance(value, dict):
                try:
                    value["default"]
                except KeyError:
                    raise ValueError("No default value at key '{}'.".format(key))

                try:
                    self[key] = random.uniform(*value["range"])
                except KeyError:
                    pass
                else:
                    continue
                try:
                    self[key] = random.choice(value["choice"])
                except KeyError:
                    raise ValueError("No 'range' or 'choice' provided at key '{}'.".format(key))

            elif isinstance(value, (int, float, str, bool)):
                pass
            else:
                raise ValueError("Invalid data type at key '{}'.".format(key))

    def set_defaults(self):
        """
        Set default values to dict.
        """
        for key, value in self.original.items():
            if isinstance(value, dict):
                try:
                    self[key] = value["default"]
                except KeyError:
                    raise ValueError("No default value at key '{}'.".format(key))
            elif isinstance(value, (int, float, str, bool)):
                pass
            else:
                raise ValueError("Invaild data type at key '{}'.".format(key))


if __name__ == '__main__':
    data = {"a": 1.232,  # No randomization
            "b": {"default": 1.234,
                  "range": [0.0, 1.1]},  # Default + uniform range
            "c": {"default": True,
                  "choice": [1, 2, 3, False, 5, True]},  # Default + choice
            "d": 1000}
    config = RandomizerDict(data)
    print(data)
    for i in range(10):
        config.randomize()
        print(config)
    config.set_defaults()
    print(config)
