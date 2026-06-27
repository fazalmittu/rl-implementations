"""

now i want to make diffusion policy from scratch

- model takes in observation and a noisy action sequence; iteratively denoises it to get a clean sequence of actions
- quick overview of DDPMs wrt images
    - there's a forward process through which we take an image (or any data) and add a bit of noise for a set number of timesteps
    - we train a model that takes in a noisy image and a diffusion timestep and it outputs how much noise was added to the original image

- ok so we need an observation encoder
    - this takes in a sequence of observations (determined by obersvation horizon)
    - if images, use ResNet to encode otherwise just standard projection layer to create an embedding
- the diffusion timestep should be represented as 1D sinusoidal embeddings 

- model finally takes in [noisy_actions (at time step t), t (sinusoidal embedding), obs embedding] -> noise
    - to be clear, this model is a U-Net in the original diffusion policy paper
    - also incorporates FiLM (which we can tackle later)
        - FiLM just applies a per-channel transformation to help the model focus on more important channel dims

- so if i wanted to start, i can start with an observation encoder
- then i can make the action denoiser model
    - U-Net kind of looks like an encoder/decoder model
    - continually downsamples image while increasing feature dim
    - then has bottleneck where most compressed/abstract info abt image is stored
    - decoder reconstructs the image
        - what helps a lot is residual connections from equivalent "levels" of the encoder part of the U-Net
        - this prevents spatial details from going missing, basically the decoder is allowed to learn smaller modifications now


obs_encoder
    images: (B, obs_horizon, num_images, C, H, W)
    state: (B, obs_horizon, state_dim)





"""