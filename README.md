# ChatBot: LLM-Powered Conversations That See and Hear Your Content

## Introduction
Discover how to build an intelligent conversation system that goes beyond text-based interactions. This project demonstrates how to enhance LLMs/vLLMs/STT models with Retrieval-Augmented Generation (RAG) techniques to create a multimodal chatbot capable of understanding and discussing your images and videos. Experience natural conversations about your content through an intuitive interface that bridges the gap between advanced AI technology and everyday visual and audio media.

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.


2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[all-docs]"
```

4. Install Tesseract for unstructured, follow guide [here](https://tesseract-ocr.github.io/tessdoc/Installation.html) for more information:

```python
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

5. We are going to use Llama 3 available on Hugging Face. Therefore, requesting the [permission](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to use it and loging in hugging face before running is required. Replace `$HUGGINGFACE_TOKEN` with your token.

```python
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN
```

## Create database
Several example data located at `data`. You can add your custom data.

Create the Chroma DB.

```python
export PYTHONPATH=$(pwd)
python backend/create_database.py
```

## Run chatbot app

```python
python app.py
```

**Please note that** the response time may vary depending on the resources available on your computer (12 GB VRAM at least).
