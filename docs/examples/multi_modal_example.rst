Multi-Modal Example Architecture
=================================

Introduction
------------

Now, verl has supported multi-modal training. You can use fsdp and 
vllm/sglang to start a multi-modal RL task. Megatron supports is also 
on the way.

Follow the steps below to quickly start a multi-modal RL task.

Step 1: Prepare dataset
-----------------------

.. code:: python

    # it will be saved in the $HOME/data/geo3k folder
    # Generic multimodal preprocessing is not shipped on exp/iclr.

Step 2: Download Model
----------------------

.. code:: bash

    # download the model from huggingface
    python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

Step 3: Perform GRPO training with multi-modal model on Geo3K Dataset
---------------------------------------------------------------------

.. code:: bash

    # run the task
    # Generic multimodal launchers are not shipped on exp/iclr.







