"""

i want to create my own ACT architecture, i don't give a fuck how long it takes

things i need / scratchpad:
- what do we start with?
    - we start with images + proprioceptive state (joint angles / gripper position)
    - images first get processed using a pretrained ResNet18 model (ImageNet)
        - outputs get resized into HIDDEN_DIM_SIZE * # TOKENS
        - concatenate values for all images
    - as for the robot state, we have a vector of size 6
        - a single linear layer will project this into a vector of size HIDDEN_DIM_SIZE

- once we have all those inputs together, we can work on the observation encoder (a transformer encoder)
    - pass in all image tokens + the proprio token
    - normal transformer model with mutli-head self attention + residuals + FFN
    - outputs a vector of same size but embeddings are a lot more "rich" with encoded info about relationships
      between parts of images / robot state

- now comes the VAE (this is where we actually pass in the data)
    - uses a transformer encoder for the encoder (to leverage attention operation)
    - the data that we are working with is actual trajectories which are just a sequence of actions
    - we want our encoder to be able to encode this entire sequence of actions in to a latent vector that represents the trajectory
    - to do so, our encoder needs to understand how the actions actually correspond to each other
    - actual flow of data
        - start by projecting each action (6D vector) to a HIDDEN_DIM_SIZE vector
        - add 1D positional encoding to each action
        - now prepend a CLS token to the list of action tokens; we have [CLS, a_0, a_1, ..., a_99]
        - now regular multi-head self attention + residual + FFN
        - output is same size as input, take the CLS token's output specifically
        - VAE has 2 heads, layers that project the CLS output to a smaller dim (latent_dim)
        - output is mean, std which paramaterize a distribution from which we can sample a latent vector

- now for the decoder 
    - start with processing inputs to get ready for the transformer
        - takes in latent vector + output of observation encoder
        - project latent vector to HIDDEN_DIM_SIZE
        - create a query embedding for each action in the chunk (so 100)
            - learned params that evolve over time
        - final inputs for transformer are: latent embedding + query embeddings + output of observation encoder
    - self attention between query embeddings / latent embedding
    - cross attention between query embeddings / observation encoder output (which are embeddings for each action in the chunk with rich features/relationships)
    - then after all decoder blocks, pass each query output (attention output) through a linear projection layer to make it a 6D action vector
    - total loss = |a_pred - a_real| + BETA * KL(q(z|a) || N(0,I))

"""