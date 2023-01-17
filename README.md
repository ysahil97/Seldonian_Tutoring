## Contents

This repository contains the code and the project report for the Seldonian Toolkit library on the Tutoring assignment problem which is modelled as a offline contextual bandit.


#### Requirements and Installation

Python3 is used in simulating these experiments.

The two main libraries required for this seldonian experiment are `seldonian-engine` and `seldonian-experiments`. Check this [link](https://seldonian.cs.umass.edu/Tutorials/tutorials/install_toolkit_tutorial/) for more details on the installation of these libraries

#### Description

The main application of the Seldonian Algorithm is in `Robinhood_SA.py`, which also generates the gradient descent plots for the Seldonian Algorithm. For further analysis of the solutions returned by the Seldonian Algorithm, its implementation is in `Robinhood_SA_experiments.py`. These python scripts are run as follows:

```
python3 Robinhood_SA.py
python3 Robinhood_SA_experiments.py <name_of_spec_file>
```

where `<name_of_spec_file>` denotes the spec file generated by `Robinhood_SA.py` to be analyzed in experiments.
The `Robinhood_SA_experiments.py` file generates the plots which showcases the trend of Performance, Solution Rate and Failure rate as the size of the dataset progresses.

#### Dataset used.
We use the intelligent tutoring dataset, which was specially curated for the Robinhood algorithm developed by Metevier et al.  We computer the initial state and reward probabilities and encode them in `Robinhood_tutoring` class present in `Environment/RL_environment.py` module.
