# KrishiGyaan Crop Recommendation System

## Introduction

### Project Overview

The KrishiGyaan Crop Recommendation System is a web application powered by machine learning, aimed at assisting farmers in choosing the best crops for their fields. By analyzing local soil properties and weather conditions, the system offers precise crop recommendations, enhancing agricultural efficiency and supporting sustainable farming practices.

## Key Features

- Accurate crop recommendations using machine learning
- User-friendly web interface for easy access
- Scalable and deployable on cloud platforms
- Supports multiple machine learning algorithms

## Model Development

### Algorithms Used

- **Decision Tree**: Breaks down data into smaller subsets by evaluating feature values, forming a tree-like model to provide decision-making clarity.
- **Gaussian Naive Bayes**: Utilizes Bayes' theorem, assuming feature independence given the class, often yielding strong results with complex, high-dimensional data.
- **Support Vector Machine (SVM)**: Identifies the optimal hyperplane that separates different classes, particularly useful for handling non-linear separations.
- **Random Forest**: Combines multiple decision trees into a single model to provide more accurate and stable predictions, particularly effective with high-dimensional datasets.

## Methodology

### System Architecture

The system architecture consists of two key elements: the web server and the ML container. The web server interacts with a database to manage data storage and retrieval, ensuring the system can scale and handle a high volume of requests efficiently.

| System Architecture |
|---------------------|
| ![image]([https://ibb.co/6D28RgR](https://i.ibb.co/6D28RgR/image.jpg)) |


### Datasets

The dataset contains 2200 entries and 8 attributes, capturing key environmental and soil factors such as nitrogen, phosphorus, and potassium (NPK) levels, temperature, humidity, pH, rainfall, and the corresponding crop label. By studying the correlations between these features and the target crop, we build a predictive model to suggest the most suitable crops based on specific soil and environmental conditions.


| Random data from the dataset for crop recommendation |
|------------------------------------------------------|
| ![image](https://ibb.co/q042XQz) |


### Crop Recommendation

We trained the dataset using four well-established machine learning algorithms: Random Forest, Gaussian Naive Bayes, Support Vector Machine (SVM), and Decision Tree. These models excel at classification tasks on labeled data, with Random Forest proving to be the most accurate for predicting suitable crops.

### Analysis Tools and Technologies

- **Python**: A flexible programming language widely used for machine learning applications.
- **NumPy**: A library that supports the creation and manipulation of large, multi-dimensional arrays and matrices.
- **Pandas**: A robust library designed for data manipulation and analysis.
- **Streamlit**: A framework used to create interactive web apps with ease.
- **HTML, CSS, JavaScript**: Core web technologies used for developing and styling the web application.

## Results

### Model Performance

The Random Forest algorithm delivered the highest performance, with an accuracy of 99.55%. A visual comparison of the accuracy across different algorithms is included for reference.

| Accuracy Comparison |
|---------------------|
| ![image](https://ibb.co/DpSvW2R) |


### Website Deployment

Built with Streamlit, this web application seamlessly integrates a machine learning engine to provide crop recommendations based on user-provided data. Its cloud-ready design ensures easy scalability and broad access, allowing it to cater to a wide range of users across various platforms.

| Landing page of the website |
|-----------------------------|
| ![image](https://ibb.co/h1kpqr8) |

| Result of crop recommendation |
|-----------------------------|
| ![image](https://ibb.co/GRKczjZ) |

---

