# RECAP: Robust Encryption for Creative Artwork Protection

This repository contains the code and supplementary materials for the paper *RECAP: Robust Encryption for Creative Artwork Protection* by Sadhana Lolla, Hannah Kim, and Christine Tu. The project introduces RECAP, a novel cryptographic approach to safeguarding artists' styles from misuse by generative AI models.

## Abstract

The rise of style mimicry models has posed significant challenges for artists, leading to tools like Glaze, a style-transfer based encryption algorithm. While effective, these tools have limitations due to secrecy and fixed secret keys. RECAP addresses these gaps with a public, robust encryption algorithm that perturbs artwork minimally in the visual domain while protecting it against generative AI models.

**Key contributions:**
1. A novel encryption-decryption pipeline preserving visual features while obfuscating style.
2. Demonstration of vulnerabilities in existing style-masking methods via adversarial attacks.
3. Evaluation using diffusion models and user surveys to validate effectiveness.

## Features

- **Encryption**: Encrypts artwork by applying minimal perturbations in the style space.
- **Decryption**: Recovers the original artwork given the encrypted image and its style key.
- **Attacks**: Neural network adversary to evaluate the robustness of the encryption.
- **Evaluation**: Pipelines to test encryption against style mimicry models and gather user feedback.

## Repository Structure
├── encryption/          # Code for the RECAP encryption algorithm
├── decryption/          # Code for the RECAP decryption algorithm
├── style_library/       # Sample datasets and style libraries
├── encrypted_data/      # Sample encrypted data to run decryption algorithm
├── images/              # Data used for generating paper figures
└── README.md            # Project overview and instructions

