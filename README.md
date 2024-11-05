# Vimy Chatbot

## Overview

The Vimy Chatbot is a conversational AI designed to assist users in studying Vim by answering questions and providing relevant information. This project is a personal take on a study chatbot, inspired by Patrick Loeber's tutorials available [here](https://www.youtube.com/playlist?list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg). The chatbot utilizes natural language processing (NLP) techniques to understand user queries and generate appropriate responses.

## Features

- **Natural Language Understanding**: The bot can interpret user input through tokenization, which splits sentences into individual words and components.
- **Stemming**: The chatbot reduces words to their root forms, allowing it to recognize different variations of the same word. For example, "organize", "organizes", and "organizing" all become "organ".
- **Bag-of-Words Model**: The chatbot utilizes a bag-of-words approach to represent user input as binary vectors, indicating the presence of known words in a given sentence. This helps the model to understand user queries regardless of their phrasing.
- **Response Generation**: It provides predefined responses based on user intents identified from their input.
- **Customizable Intents**: Users can easily modify or extend the intents and responses through a JSON file.

## Technologies Used

- **Python**: The primary programming language for developing the chatbot.
- **PyTorch**: A deep learning framework used for building and training the neural network model.
- **NLTK**: The Natural Language Toolkit, a library for working with human language data, which is used for tokenization and stemming.

## Project Structure

- `model.py`: Contains the neural network model definition and the functions for tokenization, stemming, and creating bag-of-words representations.
- `nltk_utils.py`: Provides utility functions to assist with NLP tasks (if separated).
- `intents.json`: A JSON file that holds the intents and corresponding patterns and responses that the chatbot can understand.
- `data.pth`: The saved model state and metadata after training.

