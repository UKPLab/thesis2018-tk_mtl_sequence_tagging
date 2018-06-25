"""
Module of the `TrialGenerator` class.
The trial generator takes a list of intervals and returns a randomly sampled trial from these intervals.
"""

import numpy as np


INTERVAL_TYPE_LIST = "list"
INTERVAL_TYPE_DISCRETE = "discrete"
INTERVAL_TYPE_CONTINUOUS = "continuous"


class Interval(object):
    def __init__(self, name, interval_type="list", values=None, start=None, end=None):
        """
        Initialize an interval. This is just a data-transfer object (DTO).
        It is possible to define different types of intervals:
            * list: Interval consists of a list of pre-defined values. Requires `values` parameter.
            * discrete: Interval consists of a range of integers. Requires `start` and `end` parameter.
            * continuous: Interval consists of a range of real values. Requires `start` and `end`.

        NOTE:
            * "discrete" samples in [start, end] (closed interval)
            * "continuous" samples in [start, end) (half-open interval)

        Args:
            name (str): Name of the interval. Usually, the name of the corresponding parameter.
            interval_type (str, optional): Type of interval. One of the following: list, discrete, continuous. Defaults
                to "list".
            values (list, optional): A list of values to sample from. Required for "list" type, ignored otherwise.
            start (int or float, optional): Start of the interval. Ignored for "list" type, required otherwise.
            end (int or float, optional): End of the interval. Ignored for "list" type, required otherwise.
        """
        assert isinstance(name, str)
        assert interval_type in [INTERVAL_TYPE_LIST, INTERVAL_TYPE_DISCRETE, INTERVAL_TYPE_CONTINUOUS]
        assert interval_type != INTERVAL_TYPE_LIST or isinstance(values, list), "Expected a list of values for " \
                                                                                "interval type 'list'."
        assert interval_type == INTERVAL_TYPE_LIST or (start is not None and end is not None), "Expected start and " \
                                                                                               "end to be set if the " \
                                                                                               "interval type is not " \
                                                                                               "list."
        assert interval_type == INTERVAL_TYPE_LIST or start <= end, "Start of interval should be less than end of " \
                                                                    "the interval."

        self._name = name
        self._interval_type = interval_type
        self._values = values
        if interval_type != INTERVAL_TYPE_LIST:
            self._start = int(start) if interval_type == INTERVAL_TYPE_DISCRETE else float(start)
            self._end = int(end) if interval_type == INTERVAL_TYPE_DISCRETE else float(end)

    def sample(self):
        """
        Generate a sample from the interval in a uniform way.

        Returns:
            int or float or object: Sample
        """
        if self._interval_type == INTERVAL_TYPE_LIST:
            return self._values[np.random.randint(0, len(self._values))]
        elif self._interval_type == INTERVAL_TYPE_DISCRETE:
            return np.random.random_integers(self._start, self._end).item()
        elif self._interval_type == INTERVAL_TYPE_CONTINUOUS:
            return np.random.uniform(self._start, self._end)
        else:
            raise ValueError("Invalid interval type: %s. Support only for %s." % (
                self._interval_type,
                [INTERVAL_TYPE_LIST, INTERVAL_TYPE_DISCRETE, INTERVAL_TYPE_CONTINUOUS],
            ))

    @property
    def name(self):
        return self._name


class TrialGenerator(object):
    def __init__(self, parameter_space):
        """
        Initialize the trial generator
        Args:
            parameter_space (`list` of Interval): A list of intervals.
        """
        assert all(isinstance(interval, Interval) for interval in parameter_space)

        self._parameter_space = parameter_space

    def __iter__(self):
        return self

    def __next__(self):
        """
        Python 3 compatibility
        Just calls `self.next()` internally.

        Returns:
            dict: Trial
        """
        return self.next()

    def next(self):
        """
        Generate a new trial by randomly sampling for each dimension of the parameter space.
        Returns:
            dict: Trial
        """
        return {
            interval.name: interval.sample() for interval in self._parameter_space
        }

    @property
    def parameter_space(self):
        """dict: A mapping from parameter names to value ranges"""
        return self._parameter_space


if __name__ == "__main__":

    trial_generator = TrialGenerator([
        Interval("1", INTERVAL_TYPE_DISCRETE, start=5, end=10),
        Interval("2", INTERVAL_TYPE_CONTINUOUS, start=5, end=10),
        Interval("3", INTERVAL_TYPE_LIST, values=["a", "b", "c", "d", "e"]),
        Interval("4", INTERVAL_TYPE_LIST, values=[{"x": 1}, {"x": 2}]),
    ])

    for trial in trial_generator:
        print trial
