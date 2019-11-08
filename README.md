# loadGlove
Efficiently load the pre-trained Glove embeddings or similar [pre-trained Glove embedding](https://nlp.stanford.edu/projects/glove/) files, with commonly used functions implemented.  
The following functions are supported:  
  * the embedding vector given the target word  
  * similarity between two given words  
  * top n most similar words given the target word  
  * top n most similar embedding with an embedding in list, np.ndarray, or torch.Tensor type  
  
### Usage
* To use the load_glove scripts, put the load_glove.py within the same directory of glove embedding file.  
```from load_glove import Glove

model = Glove()
model.load_model(file=path_to_file, dim=number_of_embedding_dimensions, normalize=boolean)
```  
  the default file, with no argument passed in, is to load glove.840B.300d.txt, 300 dimension, and without normalizing the embedding vectors.  
  

* To obtain number of embeddings and vector dimensions loaded by the model  
  `print(model)`
  
* To obtain embedding of a given word  
  `model.get_vector("cat")`
  if word is not in the model, the get_vector() function will return None  
  
* To obtain similarity of two given words (cosine similarity)  
  `model.similarity("cat", "dog")`
  if words are not in the model, the similarity() function will throw an exception  

* To obtain top n most simliar words with a given word (default top n is set to 10)  
  `model.most_similar("cat", topn=20)`
  
* To obtain top n most similar words with a given embedding (default top n is set to 10)  
  `model.most_similar_embedding([an embedding vector])`
  input to the most_similar_embedding() function can be an np array, a list, or a torch tensor  
