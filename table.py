import pandas as pd

# Sample data
data = {
    'Year': [1957, 1969, 1986, 2012, 2018, 2021],
    'Milestone': ['Perceptron', 'Backpropagation', 'Deep Learning', 'Deep Convolutional Networks', 'Generative Adversarial Networks', 'Transformers'],
    'Description': ['Single-layer neural network model', 'Training algorithm for multi-layer neural networks', 'Introduction of deep neural networks', 'Deep networks specifically designed for image analysis', 'Generative models for data generation', 'Transformer models for natural language processing and other tasks'],
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the table
print(df)
