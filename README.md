# Comparing Baseline vs. Fine-Tuned Models for Semantic Relatedness in African Languages

# 1. Project Overview.
The objective of this current project is to explore semantic relatedness in three African languages Hausa (hau), Kinyarwanda (kin) and Amharic (amh). The goal is to assess how multilingual baseline models perform against a fine-tuned transformer model (XLM-RoBERTa) on the Semantic relatedness task.
The set of solutions comprises of a zero-shot semantic similarity model; a fine-tuned transformer model applied transfer learning, complete execution, evaluation, and results-saving experiments.

# 2. Research Focus & Methodology.
2.1 Research Questions.
1.	How do baseline multilingual models (like LaBSE and MPNet) compare to a fine-tuned model (XLM-RoBERTa) for this task?
2.	Which transfer learning approach is best for Hausa, Kinyarwanda, and Amharic: zero-shot or fine-tuning?


# 2.2 Methodological Overview.
The study employs two modelling paradigms.

Baseline (Zero-Shot) Models.
We use the below sentence-embedding models without task-specific fine-tuning.
•	LaBSE: https://huggingface.co/sentence-transformers/LaBSE.
•	Multilingual MPNet Base V2 Rewriter: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2.
Embeddings are taken and cosine similarity is used to compute similarity.
Transfer Learning Model.
•	XLM-RoBERTa Base: https://huggingface.co/xlm-roberta-base.
A regression head was added to the model to fine-tune it on SemRel2024.
Dataset.
SemRel / SemRel2024 Dataset.
https://huggingface.co/datasets/SemRel/SemRel2024

Characteristics.
•	Semantic relatedness dataset annotated in multiple languages with the help of human.
•	Includes a number of varieties, such as Hausa, Amharic, Kinyarwanda, Zulu etc.
•	It has sentence pairs with cosine similarity scores between 0 and 1. It has train/dev/test splits.

Evaluation Metrics.
•	The primary metric for semantic similarity ranking is Spearman’s Rank Correlation (ρ).
•	Pearson Correlation.
•	Mean Absolute Error (MAE).
•	Mean Squared Error (MSE).
•	Root Mean Square Error (RMSE).

# 2.3. Environment Configuration and Dependencies.
•	Install core dependencies.
•	pip install datasets sentence-transformers transformers.
•	Use pip to install torch and more.
•	pip install matplotlib seaborn.

Optional Weights & Biases (W&B).
•	pip install wandb.

# 3. System Architecture.
The design in question employs a modular architecture to enhance its reproducibility.

Data Processing.
SemRelDataLoader.
•	Get the SemRel2024 dataset from Hugging Face.
•	Extracts language-specific subsets.
•	Performs preprocessing.
•	Creates splits for training and validation

Model Components.
•	The BaselineModel is based on cosine similarity and uses LaBSE and MPNet.
•	XLM-RoBERTa with regression output – SemanticRelatednessModel
•	SemRelDataset is a PyTorch dataset for multilingual tokenized/detokenized text inputs.

Training and Experiment Execution.
•	The term SemRelTrainer refers to fine-tuning with the Hugging Face Trainer.
•	ExperimentsExecutor-automatically run baseline, fine-tune & transfer learning
•	the main function runs the pipeline from beginning to end.

Evaluation and Analysis.
•	The Evaluation Metrics help compute correlation and error metrics
•	The ErrorAnalyzer gets sentence pairs that have high prediction errors.
•	ResultsVisualizer creates charts for comparisons and heat maps.


# 4.Optional. Real-Time Tracking (Weights & Biases).
The notebook runs fully without W&B, and a user can easily enable experiment tracking on their own accounts.
How to Enable W&B (Optional).
1.	Create an account at: https://wandb.ai.
2.	Retrieve your personal API key from your W&B profile settings.
3.	Log in via terminal:.
4.	wandb login YOUR_API_KEY.
In the notebook or Python script, set the flag to.
USE_WANDB = True.
It will turn on the W&B experiment logging, dashboards, and visualizations right away.

Optional: Real-Time Tracking (W&B).
If you enable W&B, you’ll get: - Loss curves, per epoch - Dashboards for comparing models - Saved model artifacts - Tables with predictions vs. true scores. 
You can run or reproduce results without using the optional feature. 


