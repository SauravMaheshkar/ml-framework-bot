# ML Framework Bot

An agentic workflow to translate machine learning codebases across ML frameworks reliably at scale.

## Installation

```bash
pip install -U pip
pip install -U uv
uv sync
```

## Usage

For example usage refer to [`examples/`](./examples/)

## Workflow

```mermaid
graph TD
    A[Source Snippet]
    B[Target Framework]
    
    subgraph " "
        C[Framework Identification Agent]
        D[Op Extraction Agent]
        G[Documentation Retrieval Agent]
    end

    subgraph " "
        H[Source Documentation Chunks]
        I[Reference Documentation Retrieval Agent]
        J[Relevant Documentation Chunks<br>for Target Framework]
    end

    subgraph " "
        K[Translation Agent]
        L[Translated Snippet]
    end

    %% Edges
    A --> C
    C -- Source Framework --> D
    A --> D
    D -- Operations --> G
    G --> H
    H --> I
    B --> I
    I --> J
    H --> K
    J --> K
    A --> K
    K --> L
```
