## Tabular NN Testing

#### Goals

Nothing exciting going on here, just testing ideas from various papers for tabular neural networks.

The main data source is ATP tennis match results from [here](https://github.com/JeffSackmann/tennis_atp).

#### What's in here?

 - `data_construction.py` is dedicated to preprocessing data from the aforementioned tennis data sources. It constructs a few files that the network training notebook uses.
 - `dev_notebooks/toy_example.ipynb` constructs a synthetic data set and then trains a model on it. It works pretty well, but I don't think it's a very hard problem or anything. I just wanted to see if it works.
 - `dev_notebooks/network_testing.ipynb` is the real thing, trying to learn to predict tennis match results. Currently the data is not very rich, but it took a while to get the network to learn anything anyway. Eventually the goal is a combined tabular/sequential data set, but not at the moment.
 - `hybrid-training.py` trains a network on combined sequential and tabular data. **This is hopefully the interesting part.**
 - `wn/data` contains a dataset class and some preprocessing functions. `DataInterface` is probably the interesting class here; it streamlines defining and collecting some information for preprocessing features. Both the ugly `tr` function (which implements some preprocessing) and the `TabularInputLayer` class use a `DataInterface` to streamline setup somewhat.
 - `wn/net` contains network-related code. It has the classic "move a dictionary to a device in one call" function `to_`, an implementation of Time2Vec (which is hopefully right) and a couple modules to simplify building the whole network, together with a generic Torch `TransformerEncoder`. More interesting is the `TabularInputLayer`, which defines different kinds of input layers for each feature in the table, depending on its type, as well as adding a `<CLS>` token and appending learned column encodings.

There's normal `.gitignore` and `requirements.txt` too, but the requirements especially are a little vague at the moment.

#### How to run this?

In the project directory, something like:

  - `pip install -r requirements.txt`
    - You can remove mlflow if you don't intend to use it.
  - `pip install -e .`
    - This will install the wn module (in editable mode) so you can use it no matter where your script is located.
  - `mkdir data`
    - The data construction script wants to write here.
  - `python data_construction.py`
    - This preprocesses and saves the tennis data. It should hopefully be fast, but it's not super optimized.
  - `python hybrid_training.py`
    - This trains the network.

#### References

Er, I'll document these later. There are a lot.
