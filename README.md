# IoT-IDS-VAE-Classification

**View write up here**: https://connormcmanigal.github.io/IoT-IDS-VAE-Classification/Final_Report.pdf

*Connor McManigal, Aaron Mui, Rishabh Verma, and Peyton Politewicz*

## Abstract:

We investigate the performance of two different variational autoencoder (VAE) approaches to the ever-expanding space of utilizating machine learning for network traffic monitoring and safety. We train a baseline VAE, and then compare its performance to a Mixed-loss VAE (MLVAE), testing if its loss structure of separating categorical and continuous features offer improved performance in this use case. We chose VAEs because the learned latent structure of the data is particularly useful when labels are sparse and difficult to obtain, which is often the case for network attacks. For the same reason, they often generalize well on unseen data and are scalable. We train these VAEs on normal, expected traffic, and then leverage the variety of attack data to validate and test our filtration approach. We utilize the RT-IoT2022 dataset, sourced from the UC Irvine Machine Learning Repository, whose contents are intended to simulate local area network communication between internet-connected 'smart' devices, packaged with simulated attack traffic. Compared to other related works, we add a second layer to classify specific varieties of network traffic, instead of detecting anomalies and sorting data into 'attack'/'normal' traffic patterns by reconstruction error.
