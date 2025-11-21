African Languages Semantic Relatedness with W&B Integration
1. Project Overview
This repository contains a complete pipeline for analyzing Semantic Relatedness (SemRel) in three African languages—Hausa (hau), Kinyarwanda (kin), and Amharic (amh). The goal is to compare multilingual models with a fine-tuned transformer model on this task.
All experiments are meticulously tracked using Weights & Biases (W&B) for real-time performance monitoring, metric logging, and artifact management.
2. Research Focus & Methodology
The project addresses two primary research questions:
Transfer Learning Efficacy: Which transfer learning approaches best improve semantic relatedness determination in African languages?
Baseline Comparison: How do baseline multilingual models (LaBSE, MPNet) compare to the fine-tuned XLM-RoBERTa model for Hausa, Kinyarwanda, and Amharic?
Methodology Summary:
The experiments are structured around the following components:
Baseline Models: We use LaBSE and the Multilingual MPNet (paraphrase-multilingual-mpnet-base-v2). These models are used for zero-shot evaluation.
Transfer Learning Model: We fine-tune the XLM-RoBERTa model (xlm-roberta-base) on the combined SemRel training data.
Data Source: All models are evaluated on the SemRel/SemRel2024 dataset from Hugging Face.
Evaluation Metrics: The primary performance indicator is the Spearman Correlation ($\rho$), supported by the Mean Absolute Error (MAE) and Pearson Correlation.
3. Setup and Prerequisites
3.1. Package List
The project relies on standard NLP and machine learning libraries. You can install all necessary dependencies using the following commands:

Install Hugging Face datasets and sentence-transformers
pip install datasets sentence-transformers transformers
Install W&B for tracking
pip install wandb
Install general ML/data processing libraries
pip install torch numpy pandas scikit-learn scipy matplotlib seaborn


3.2. Weights & Biases (W&B) Setup
To enable experiment tracking, you must have a W&B account and be logged in:
Sign up/Log in: Create an account at wandb.ai.
Login via Terminal: Run the following command and enter your API key when prompted:
wandb login


4. How to Run Experiments
The entire pipeline is contained within the main Python file.
4.1. Execution
Execute the script directly from your terminal:
python u25606426_miracle_zvirikuzhe_final_project_cos80.py


4.2. Configuration
By default, the script runs the full pipeline, including W&B tracking. You can disable W&B by changing the boolean flag at the bottom of the script:

In u25606426_miracle_zvirikuzhe_final_project_cos80 (around line 700)
if __name__ == "__main__":
    Set to False if you don't have W&B configured
    USE_WANDB = True # Change this to False to run without W&B
    results = main(use_wandb=USE_WANDB)


The key hyperparameters for the fine-tuning phase are set in the main function's config dictionary and are logged to W&B:
config = {
    'languages': ['hau', 'kin', 'amh'],
    'epochs': 2,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'val_size': 0.2,
    'random_state': 42,
    'model_name': 'xlm-roberta-base'
}


4.3. Outputs
Upon completion, the script will:
Print the experiment results table to the console.
Save the complete results to a CSV file in the ./results directory:
./results/experiment_results.csv
5. Pipeline Structure and Documentation
The code is organized into logical classes and functional sections to maximize clarity and maintainability.
Setup: The WandBTracker utility handles initializing, logging metrics, and finishing the Weights & Biases run.
Data Handling: The SemRelDataLoader manages loading and preprocessing the SemRel/SemRel2024 dataset, including creating train/validation splits.

Models:
BaselineModel encapsulates the inference logic for the LaBSE and MPNet sentence embedding models.
SemanticRelatednessModel defines the regression head added on top of the transformer backbone (XLM-RoBERTa) for fine-tuning.
SemRelDataset is a PyTorch Dataset used for tokenizing sentence pairs and preparing scores for the fine-tuning process.
Training & Execution:
SemRelTrainer wraps the Hugging Face Trainer instance, integrating the model fine-tuning with W&B tracking.
ExperimentRunner acts as the orchestrator, managing the sequential execution of both baseline and transfer learning experiments.
main() is the entry point that executes the entire pipeline end-to-end.

Evaluation & Analysis:
EvaluationMetrics calculates all the necessary metrics, including Spearman, Pearson, MAE, MSE, and RMSE.
ErrorAnalyzer generates detailed reports on prediction errors, focusing on worst-case analysis and performance across different score ranges.
ResultsVisualizer creates plots (comparison charts, heatmaps) for the metrics and logs them as W&B Images.

6. Real-time Tracking (W&B)
The project heavily utilizes Weights & Biases to track and compare all experiments:

