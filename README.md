# loadGlove
Efficiently load the pre-trained Glove embeddings or similar [pre-trained Glove embedding](https://nlp.stanford.edu/projects/glove/) files, with commonly used functions implemented.  
The following functions are supported:
  * the embedding vector given the target word  
  * similarity between two given words  
  * top n most similar words given the target word  
  * top n most similar embedding with an embedding in list, np.ndarray, or torch.Tensor type  
  
### Usage
* To use the load_glove scripts, put the load_glove.py within the same directory of glove embedding file.  
  ```
  from load_glove import Glove
  
  model = Glove()
  model.load_model(file=path_to_file(str), dim=embedding_dimensions(int), normalize=boolean)
  ```  
  By default, with no argument passed in, the load_model() function will run with the following parameters: `load_model(file="./data/glove.840B.300d.txt", dim=300, normalize=False)`  


* To obtain number of embeddings and vector dimensions loaded by the model  
  `print(model)`
  
* To obtain embedding of a given word. If word is not in the model, the get_vector() function will return None    
  `model.get_vector("cat")`  
  
* To obtain similarity of two given words (cosine similarity). iI words are not in the model, the similarity() function will throw an exception  
  `model.similarity("cat", "dog")`  

* To obtain top n most simliar words with a given word (default top n is set to 10)  
  `model.most_similar("cat", topn=10)`  
  
* To obtain top n most similar words with a given embedding (default top n is set to 10). Input to the most_similar_embedding() function can be an np array, a list, or a torch tensor  
  `model.most_similar_embedding([an embedding vector], topn=10)`  
