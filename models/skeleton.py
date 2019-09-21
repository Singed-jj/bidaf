
import tensorflow as tf
import numpy as np


class Skeleton:

    def character_embedding_layer(self):
        """
        character embedding layer
        k_d = 100dimension
        """
        pass

    def word_embedding_layer(self):
        """
        word embedding layer

        w_d = 300dimension
        concatenation of word_embedding_layer and character_embedding_layer  d_d = k_d + w_d
        """
        pass

    def contextual_embedding_layer(self):
        """
        contextual embedding layer

        bi-directional LSTM
        """
        pass

    def attention_flow_layer(self):
        """
        attention flow layer

        1. context to query attention
        2. query to context attetion
        """

    def Modeling_layer(self):
        """
        modeling layer

        bi-directional LSTM

        """

    def bi_directional_lstm(self):
        """

        """
