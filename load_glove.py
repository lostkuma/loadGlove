############### Copyright By Shannon Jing ################
######### For issues contact xiaonjing@gmail.com #########

FILENAME = "./data/glove.840B.300d.txt"
DIMENSION = 300
import numpy as np
import torch

def count_lines(file_obj):
    """ count num of rows in file obj """
    for idx, line in enumerate(file_obj):
        pass
    file_obj.seek(0)
    return idx + 1

class Glove(object):
    """ load golve vectors in as one np matrix.
        load texts in to an array"""
    def __init__(self):
        self.embeddings_mat = None # np matrix for vectors
        self.tokens_arr = None # np array for texts
        self.token_to_idx = dict() # hash map for texts to index in array/matrix
        self.num_tokens = 0 # int value to store vocab size
        self.dimension = None # save vector dimension
        self.normalize = None # save a bool for if normalize when loading the vectors

    def __repr__(self):
        """ print Glove obj with vocab size and dim """
        return "<Glove_object num_tokens.{} vec_dim.{}>".format(self.num_tokens, self.dimension)

    def load_model(self, file=FILENAME, dim=DIMENSION, normalize=False):
        """ load pretrained embedding from file 
            each row of file must have format: text dim1 dim2 ... """
        print("Loading pretrained Glove vectors from file {}".format(FILENAME))
        self.dimension = dim
        self.normalize = normalize
        with open(file, "r", encoding="utf-8") as textfile:
            self.num_tokens = count_lines(textfile)
            self.tokens_arr = ["" for i in range(self.num_tokens)]
            self.embeddings_mat = np.zeros((self.num_tokens, self.dimension))

            for idx, line in enumerate(textfile):
                line = line.split()
                token = ''.join(line[:-self.dimension])
                self.tokens_arr[idx] = token
                self.token_to_idx[token] = idx 
                vec = list(map(float, line[-self.dimension:]))
                if self.normalize: 
                    # normalize the vectors as they are put into the matrix
                    vec = vec / np.linalg.norm(vec)
                self.embeddings_mat[idx] = vec 
                if (idx+1) % 200000 == 0:
                    print("  --{}%  loaded.".format(round(idx/self.num_tokens*100, 2)))
        print("Finished loading Glove model. {} vectors loaded".format(self.num_tokens))

    def get_vector(self, token):
        """ input a string token (that is in the model) 
            return the embedding given a token input
            return None if token is not in the model """
        try:
            idx = self.token_to_idx[token]
        except KeyError:
            print("Input token <{}> is not in the model. Will return None type vector".format(token))
            return None
        return self.embeddings_mat[idx]

    def similarity(self, token1, token2):
        """ input two string tokens (that are in the model)
            return cosine similarity (dot product) between the corresponding embeddings """
        vec1 = self.get_vector(token1)
        vec2 = self.get_vector(token2)
        assert vec1 is not None and vec2 is not None, "Cannot compute similarity between None type vectors."
        if not self.normalize:
            # if model not loaded as normalized embeddings 
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
        return np.dot(vec1, vec2)

    def most_similar(self, token, topn=10):
        """ return the most similar token and the cosine similarity with the input token
            default return 10 most similar tokens
            return format in list of tuples """
        vec = self.get_vector(token)
        assert vec is not None, "Cannot compute similarity between None type vectors."
        return self.most_similar_embedding(vec, topn+1)[1:]

    def most_similar_embedding(self, embedding, topn=10):
        """ input type: np.ndarray, torch.Tensor, or list 
            output topn most similar tokens and cosine similarity (dot product) """
        assert type(embedding) in [np.ndarray, list, torch.Tensor], "Input type must be np.array, list, or torch.Tensor."   
        vec = np.asarray(embedding) # convert to array
        vec = vec / np.linalg.norm(vec) # normalize vec
        if self.normalize:
            dot_mat = np.sum(np.multiply(vec, self.embeddings_mat), axis=1) # dot product alone the rows
        else:
            dot_mat = np.sum(np.multiply(vec, self.embeddings_mat / np.linalg.norm(self.embeddings_mat, axis=1)[:, None]), axis=1) # normalize and dot product alone the rows
        assert len(dot_mat) == self.num_tokens, "Error in computing cosine similarity, number of vocabs don't match before and after."
        topn_idx = dot_mat.argsort()[::-1][:topn] # argsort() returns the sorted idx, reversed top n
        topn_most_similar = list(zip([self.tokens_arr[x] for x in topn_idx], [dot_mat[x] for x in topn_idx])) 
        return topn_most_similar

