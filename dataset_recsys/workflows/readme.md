# Workflows

This folder contains **workflow definitions that orchestrate end-to-end processes**
in the dataset recommendation system.

A workflow coordinates multiple components of the system (ingestion, preprocessing,
embedding generation, recommendation computation, and storage) to produce
serving-ready outputs.

Typical workflow steps include:

- fetching dataset metadata
- enriching and preprocessing profiles
- generating embeddings
- storing embedding metadata and vectors
- computing recommendations
- writing results to the serving layer (e.g., Redis)

## Rule of thumb

Put code here if it **orchestrates multiple components into a complete system process**,
rather than implementing individual steps.