# Industrial Multi-Objective Experimental Design & Optimization Toolkit

A robust Python-based Design of Experiments (DoE) and process optimization system tailored for R&D and production environments. It bridges traditional space-filling experimental design with regression to handle complex, multi-objective process optimization challenges.

## Key Features

* **Advanced Space Filling**: Leverages Latin Hypercube Sampling (LHS) to ensure uniform coverage across high-dimensional design spaces.
* **Uncertainty-Aware Modeling**: Employs Gaussian Process Regression (GPR) to map nonlinear responses and evaluate prediction confidence intervals.
* **Robustness Optimization**: Automatically weighs process variance alongside mean targets to locate stable operating windows.
* **Sequential Convergence**: Implements automated stopping criteria based on factor dimensions and local variance thresholds.
* **Replicate Expansion**: Supports dynamic multi-run data expansion and mirrored point generation for accurate experimental error evaluation.
* **Built-in Safety Defenses**: Handles edge-case conditions (such as constant single-class outputs) to prevent numerical crashes during automated iterations.

## Getting Pre-Compiled Binaries

If you are using the pre-compiled builds from GitHub Actions:

### For Windows
Download the zipped executable artifact, extract it, and launch the application directly.

### For macOS
1. Extract the downloaded archive.
2. If macOS blocks the app on first launch due to missing developer signatures, go to **System Settings ➔ Privacy & Security** and click **"Open Anyway"**.
3. The app automatically launches a local web interface and opens your default browser.
