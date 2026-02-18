# Title
End-to-End Probabilistic Modeling of INRs via Neural Diffusion in Parameter Space

# Brief Description

Implicit Neural Representations (INRs) provide a powerful functional alternative to discrete representations for modeling continuous signals, and their generative formulation has recently attracted increasing interest. However, learning distributions over the parameters that define INRs remains challenging, as it typically requires computationally intensive two-stage training procedures. One line of research first learns a collection of INRs and subsequently trains a diffusion model over their vectorized parameters. A second approach infers latent variables using an encoder, employs hypernetworks to map these latent representations to INR parameters, and then trains diffusion models in the latent space.

In a separate line of work on diffusion-based data generation, Neural Diffusion Models (NDMs) have been recently proposed as an alternative framework in which a learnable encoder models the forward corruption process conditioned on the data, demonstrating improved generative performance against fixed forward diffusion trajectories.

This thesis proposes a novel approach to generative modeling of INRs by combining ideas from Neural Diffusion Models and recent Transformer-based hypernetworks. Specifically, we adopt the NDM framework and introduce a Hyper-Transformer encoder that maps observed data directly to the forward noise trajectory in parameter space, jointly learning both the trajectory and its terminal state. This design eliminates the need for the two-stage training procedures commonly used in latent diffusion models, enabling end-to-end training of a neural diffusion model directly over INR parameters.
