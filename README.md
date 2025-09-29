📌 Project Overview
The goal of this project is to understand how different countries describe, frame, and prioritize quantum technologies in their national strategy documents. Using topic modeling (BERTopic), we uncover recurring themes and narratives across more than 50 national quantum strategy documents.
The workflow covers:
- Loading and preprocessing policy text data
- Generating document embeddings with SentenceTransformers
- Applying dimensionality reduction (UMAP) and clustering (HDBSCAN)
- Extracting interpretable topics with BERTopic
- Iteratively merging and refining topics to reach a stable set of ~64 topics
- Analyzing how topics vary across countries and over time
- Exporting results for further qualitative and quantitative assessment

⚙️ Main Features
- Reproducible BERTopic Pipeline
- Prepares embeddings, reduces dimensions, clusters documents, and extracts topics.
- Custom Vectorization & Stopword Handling
- Uses a custom CountVectorizer subclass for flexible n-grams (1–3).
- Parameter Tuning with TopicModelTuner
- Supports systematic search over HDBSCAN parameters for robust clustering.
- Topic Refinement & Merging
- Iteratively merges semantically similar clusters to produce interpretable topics.
- Cross-Country & Temporal Analysis
- Topics per country (absolute and relative frequencies)
- Topics over time (yearly trends in national quantum strategies)
- Visualization & Export
- Interactive hierarchical topic maps
- Export of representative documents, topic keywords, and topic prevalence (CSV/Excel)

📂 Repository Structure
- dataframe_policy.csv – Input dataset (policy documents, metadata)
- Main script – End-to-end analysis with BERTopic
- Data loading & descriptives
- Embedding + clustering + topic extraction
- Topic merging & labeling
- Country- and time-based breakdowns
- Model saving & exporting results
Output files include:
- Topic overviews (t_topics.csv, Merged_Topics.csv)
- Representative sentences per topic
- Topics per country (topics_per_country.csv)
- Topics over time (topics_overtime.xlsx)

🚀 How to Use
- Place the dataset in the working directory (dataframe_policy.csv).
- Run the script step by step (Jupyter/Spyder recommended).
- Adjust file paths to your local machine (/Users/...).
- Explore exported CSV/Excel results and interactive visualizations.


🛠️ Dependencies
- Detailed description and information about BERT can be found here maartengr.github.io/BERTopic/ 
