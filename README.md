Overview
Accent_Identification is a deep learning-based speech classification project focused on identifying speaker accents from audio recordings using MFCC (Mel Frequency Cepstral Coefficients) features. The project explores multiple neural network architectures for sequence-based audio classification, including LSTM, Bidirectional LSTM, Attention-based models, and CNN-LSTM hybrids.
The primary goal of this project is to experiment with different deep learning approaches for multi-class accent classification and compare their performance using validation accuracy and loss metrics.
Based on the provided notebook and implementation details 
________________________________________
Features
•	Audio preprocessing and MFCC feature extraction 
•	Multi-class accent classification 
•	Data normalization and augmentation 
•	Handling class imbalance using class weights 
•	Multiple deep learning model experiments: 
o	Stacked LSTM 
o	Bidirectional LSTM 
o	BiLSTM with Attention 
o	Conv1D + BiLSTM hybrid 
•	Training and validation visualization 
•	Performance comparison across architectures 
________________________________________
Technologies Used
•	Python 
•	NumPy 
•	librosa 
•	TensorFlow / Keras 
•	scikit-learn 
•	matplotlib 
•	pydub 
________________________________________
Project Workflow
1. Audio Loading & Feature Extraction
Audio files (.wav) or precomputed MFCC .npy files are loaded and processed using librosa.
MFCC features are extracted from speech recordings to capture important speech and pronunciation characteristics.
2. Data Preprocessing
•	Label encoding for accent classes 
•	One-hot encoding for classification 
•	Train-test split with stratification 
•	Data normalization 
•	Data augmentation experiments 
•	Class imbalance handling using class weights 
3. Model Architectures Explored
The project experiments with several architectures:
Stacked LSTM
Used to capture temporal dependencies in speech sequences.
Bidirectional LSTM
Captures contextual information from both past and future speech frames.
Attention-Based BiLSTM
Introduces attention layers to focus on important temporal speech patterns.
CNN + BiLSTM Hybrid
Combines convolutional feature extraction with sequential learning.
________________________________________
Installation
pip install numpy librosa tensorflow scikit-learn matplotlib pydub
________________________________________
Dataset Structure
Example dataset structure:
Dataset/
│
├── Accent_1/
│   ├── sample1.wav
│   ├── sample2.wav
│
├── Accent_2/
│   ├── sample1.wav
│   ├── sample2.wav
Or using extracted MFCC .npy files:
MFCC_Features/
│
├── Accent_1/
│   ├── file1.npy
│
├── Accent_2/
│   ├── file2.npy
________________________________________
Running the Project
Train the Model
Run the training notebook or Python script after updating dataset paths:
root_folder = "/content/drive/MyDrive/MFCC_Features"
or
root_folder = "/content/drive/MyDrive/Removed_Pauses"
________________________________________
Example Training Process
history = model.fit(
    X_train_augmented,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_augmented, y_test)
)
________________________________________
Evaluation Metrics
The models are evaluated using:
•	Accuracy 
•	Validation Accuracy 
•	Loss 
•	Validation Loss 
Training and validation curves are visualized using matplotlib.
________________________________________
Results
Different architectures produced varying performance levels during experimentation.
Observed test accuracies in experiments ranged approximately between:
•	38% to 44% for several sequence models 
•	Higher scores in synthetic-data experiments 
This project should be considered an experimental academic deep learning project focused on architecture exploration rather than a production-ready system.
________________________________________
Key Learnings
•	Sequence models like LSTM and BiLSTM can capture temporal speech patterns from MFCC features. 
•	Attention mechanisms can be integrated into speech classification pipelines. 
•	Audio preprocessing and class balancing significantly impact model training. 
•	CNN-LSTM hybrids can improve feature learning for sequential audio data. 
________________________________________
Future Improvements
•	Use larger and cleaner accent datasets 
•	Improve augmentation strategies 
•	Hyperparameter tuning 
•	Better feature engineering 
•	Transformer-based speech models 
•	Real-time accent prediction API deployment 
________________________________________
Project Summary
This project demonstrates an end-to-end workflow for speech accent classification using deep learning. It includes audio preprocessing, MFCC extraction, sequential neural network modeling, evaluation, and experimentation across multiple architectures to study accent recognition performance.
Code and implementation details were based on the provided notebook and training experiments.

