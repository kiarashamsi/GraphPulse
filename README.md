# GraphPulse: Topological Representations for Temporal Graph Property Prediction

This repository contains the implementation of GraphPulse, a framework for temporal graph property prediction. GraphPulse combines topological data analysis with recurrent neural networks to effectively predict the evolution of temporal graphs.

## Abstract

Many real-world networks evolve over time, and predicting the evolution of such networks remains a challenging task. Graph Neural Networks (GNNs) have shown empirical success for learning on static graphs, but they lack the ability to effectively learn from nodes and edges with different timestamps. Consequently, the prediction of future properties in temporal graphs remains a relatively under-explored area.

In this paper, we aim to bridge this gap by introducing a principled framework, named GraphPulse. The framework combines two important techniques for the analysis of temporal graphs within a Newtonian framework. First, we employ the Mapper method, a key tool in topological data analysis, to extract essential clustering information from graph nodes. Next, we harness the sequential modeling capabilities of Recurrent Neural Networks (RNNs) for temporal reasoning regarding the graph’s evolution.

Through extensive experimentation, we demonstrate that our model enhances the ROC-AUC metric by 10.2% in comparison to the top-performing state-of-the-art method across various temporal networks.
