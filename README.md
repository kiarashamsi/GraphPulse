<p align="center">
  <img width="200" height="200" src="https://github.com/kiarashamsi/GraphPulse/blob/master/image/Graphpulse.png">
</p>

# GraphPulse: Topological Representations for Temporal Graph Property Prediction

This repository contains the implementation of GraphPulse, a framework for temporal graph property prediction. GraphPulse combines topological data analysis with recurrent neural networks to effectively predict the evolution of temporal graphs.

## Overview

Many real-world networks evolve over time, and predicting the evolution of such networks remains a challenging task. Graph Neural Networks (GNNs) have shown empirical success for learning on static graphs, but they lack the ability to effectively learn from nodes and edges with different timestamps. Consequently, the prediction of future properties in temporal graphs remains a relatively under-explored area.

In this project, we aim to bridge this gap by introducing a principled framework, named GraphPulse. The framework combines two important techniques for the analysis of temporal graphs within a Newtonian framework. First, we employ the Mapper method, a key tool in topological data analysis, to extract essential clustering information from graph nodes. Next, we harness the sequential modeling capabilities of Recurrent Neural Networks (RNNs) for temporal reasoning regarding the graph’s evolution.

Through extensive experimentation, we demonstrate that our model enhances the ROC-AUC metric by 10.2% in comparison to the top-performing state-of-the-art method across various temporal networks.

![](https://github.com/kiarashamsi/GraphPulse/blob/master/image/System-overview%20(4).png)
*Graphpulse system overview*



### Analyzer Package
​Our repository hosts the comprehensive analyzer package, which encompasses a suite of sequence and graph processing utilities designed for data preparation and Topological Data Analysis (TDA) sequence extraction. To get started with utilizing these functionalities, clone the repository and import the analyzer package into your project as follows:

```
git clone https://github.com/kiarashamsi/GraphPulse.git
from analyzer import <module_name>
```

The package comes fully equipped with all necessary extractor sections, making the data processing workflow seamless and efficient.

### Model Implementation
For those interested in neural network implementations, our repository contains baseline Recurrent Neural Network (RNN) and Graph Neural Network (GNN) models located within the rnn and gnn packages respectively. Additionally, we provide our novel model, GraphPulse, which can be found under the same directories for a direct comparison with the baseline implementations.

Should your research require state-of-the-art (SOA) models, these have been implemented in a separate library included within our repository for your convenience.

### Data Access
The data folder is your go-to destination within the repository to find the datasets and pre-extracted sequences. This resource is structured to facilitate easy access and to expedite the integration of data into your research workflows.

For a closer look at the available data and instructions on how to use it, please visit the data folder:

```
repository-root/
└── data/
    ├── all_network/
    └── Sequences/
```

We encourage you to explore the repository and leverage these prepared resources for your projects and research endeavors.

### Prerequisites

- Python 3.6+
- Libraries listed in `requirements.txt`

### Cite Us
```
@inproceedings{shamsi2024graphpulse,
    title={GraphPulse: Topological Representations for Temporal Graph Property Prediction},
    author={Shamsi, Kiarash and Poursafaei, Farimah and Huang, Shenyang and Ngo, Bao Tran Gia and Coskunuzer, Baris and Akcora, Cuneyt Gurcan},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```



