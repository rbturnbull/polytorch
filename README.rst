================================================================
polytorch
================================================================

.. start-badges

|pypi badge| |testing badge| |coverage badge| |docs badge| |black badge| 

.. |pypi badge| image:: https://img.shields.io/pypi/v/polytorch?color=blue
   :alt: PyPI - Version
   :target: https://pypi.org/project/polytorch/

.. |testing badge| image:: https://github.com/rbturnbull/polytorch/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/polytorch/actions

.. |docs badge| image:: https://github.com/rbturnbull/polytorch/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/polytorch
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/f84ac74436887cd42d77fbe2246d1f57/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/polytorch/coverage/
    
.. end-badges

.. start-quickstart

Embeddings and loss functions for different data types.

Installation
==================================

Requires Python 3.10 or later. Install using pip:

.. code-block:: bash

    pip install polytorch

Or install the latest version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/polytorch.git


Static image export uses Kaleido 1.x and requires Chrome or Chromium.
If no compatible browser is installed, install one with:

.. code-block:: bash

    plotly_get_chrome

Data Types
==================================

This package allows you to input and output different data types in PyTorch models.

Binary Data
----------------------------------------

.. code-block:: python

    from polytorch import BinaryData

    binary_data = BinaryData()

    # Or with labels and colors
    binary_data = BinaryData(labels=["no_feature", "with_feature"], colors=["red", "blue"])

Categorical Data
----------------------------------------

.. code-block:: python

    from polytorch import CategoricalData

    category_count = 5  # Number of categories
    categorical_data = CategoricalData(category_count)

    # Or with labels, colors and label smoothing
    categorical_data = CategoricalData(
        category_count=category_count,
        labels=["cat", "dog", "fish", "bird", "reptile"],
        colors=["red", "blue", "green", "yellow", "purple"],
        label_smoothing=0.1,
    )

Ordinal Data
----------------------------------------

.. code-block:: python

    from polytorch import OrdinalData

    ordinal_data = OrdinalData(category_count=5)
    
    # Or with color
    ordinal_data = OrdinalData(category_count=5, color="pink")
    
Continuous Data
----------------------------------------

.. code-block:: python

    from polytorch import ContinuousData

    continuous_data = ContinuousData()

    # Or with color
    continuous_data = ContinuousData(color="orange")


Hierarchical Data
----------------------------------------


.. code-block:: python

    from polytorch import HierarchicalData
    from hierarchicalsoftmax import SoftmaxNode

    root = SoftmaxNode("root")
    child1 = SoftmaxNode("child1", parent=root)
    child2 = SoftmaxNode("child2", parent=root)
    tip1 = SoftmaxNode("tip1", parent=child1)
    tip2 = SoftmaxNode("tip2", parent=child1)
    tip3 = SoftmaxNode("tip3", parent=child2)
    tip4 = SoftmaxNode("tip4", parent=child2)
    
    
    hierarchical_data = HierarchicalData(root)

Build a hierarchy from slash-separated paths with ``from_series``:

.. code-block:: python

    import pandas as pd

    hierarchical_data, node_ids = HierarchicalData.from_series(
        "taxonomy",
        pd.Series(["Animal/Mammal/Dog", "Animal/Bird/Eagle", " Animal / Mammal / Dog "]),
    )
    assert node_ids[0] == node_ids[2]

Whitespace around each path component is stripped. Shared paths reuse nodes.
Multiple top-level names are joined under a virtual ``__root__`` node.
    
Embedding your data
==================================

.. code-block:: python

    from torch import nn
    from polytorch import PolyEmbedding
    
    class MyModule(nn.Module):
        def __init__(self, embedding_size:int=128):
            super(MyModule, self).__init__()

            input_types = [binary_data, categorical_data] # for example. Could be other data types as well.
            self.embedding = PolyEmbedding( input_types=input_types, embedding_size=embedding_size)
            
            # Other modules
            ...

        def forward(self, x_binary, x_categorical):
            
            embedded = self.embedding( x_binary, x_categorical )

            # Use the embedded features in your model
            ...            

.. warning:: 

    The ``HierarchicalData`` type is not yet supported by the ``PolyEmbedding`` class.

Outputting your data
==================================

You can also get your model to output to different data types.

.. code-block:: python

    from torch import nn
    from polytorch import PolyLazyLinear

    output_types = [
        CategoricalData(category_count=5, loss_weighting=0.5),  # For example, a categorical output with 5 categories
        BinaryData(loss_weighting=1.0),                        # A binary output
        ContinuousData(loss_weighting=0.1)                     # A continuous output
    ]

    class MyModule(nn.Module):
        def __init__(self, output_types):
            super(MyModule, self).__init__()

            self.output = PolyLazyLinear(output_types=output_types)

        def forward(self, x):
            # Your model logic
            ...

            # Output to different data types
            return self.output(x)

Use these data types to configure the loss:

.. code-block:: python

    from polytorch import PolyLoss

    loss_module = PolyLoss(data_types=output_types)

    # In your training loop
    loss = loss_module(predictions, categorical_target, binary_target, continuous_target)

DataFrames
====================

Use ``PolyDataFrame`` with a pandas dataframe or a CSV file supplied as a
``pathlib.Path``. Annotate columns as ``name:type``; unannotated columns are
available by name but are omitted from row tuples and ``data_types``.

.. code-block:: python

    import pandas as pd
    from polytorch import PolyDataFrame

    frame = PolyDataFrame(pd.DataFrame({
        "id": [100, 101],
        "tree:Hierarchical": ["Animal/Dog", "Animal/Cat"],
        "score:continuous": [0.25, 0.75],
    }))
    tree_id, score = frame[0]
    assert score == 0.25
    assert frame["score", 1] == 0.75
    assert frame["id"][0] == 100
    assert [data_type.name for data_type in frame.data_types] == ["tree", "score"]

Types accept an optional ``Data`` suffix (for example, ``ContinuousData``).
Use ``data_columns={"score"}`` to select annotated columns by name. Row tuples
and ``data_types`` follow the original column order. Categorical and ordinal
values are returned unchanged; supply integer category indices for use with
embeddings and cross-entropy loss.


.. end-quickstart


Credits
==================================

.. start-credits

Robert Turnbull
For more information contact: <robert.turnbull@unimelb.edu.au>


.. end-credits

