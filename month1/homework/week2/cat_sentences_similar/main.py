import numpy as np
from scipy.spatial.distance import cosine
import re


class Sentences:
    def __init__(self):
        self._list = []

    def add(self, sentence):
        split_sentence = re.split('[^a-z]', sentence)
        split_sentence = [word for word in split_sentence if len(word) > 0]
        self._list.append(split_sentence)

    def get_dictionary(self):
        dictionary = Dictionary()
        for sentence in self._list:
            for word in sentence:
                dictionary.add(word)
        return dictionary

    def count_word_in_sentence(self, index, word):
        sentence = self._list[index]
        count = 0
        for ex_word in sentence:
            if ex_word == word:
                count += 1
        return count

    def size(self):
        return len(self._list)


class Dictionary:
    def __init__(self):
        self._index = 0
        self._dict = {}

    def add(self, word):
        if not self.has(word):
            self._dict[self._index] = word
            self._index += 1

    def has(self, word):
        for index, ex_word in self._dict.iteritems():
            if ex_word == word:
                return True
        return False

    def as_dict(self):
        return self._dict


file_obj = open('sentences.txt', 'r')

sentences = Sentences()
for line in file_obj:
    line = line.strip().lower()
    sentences.add(line)

word_dictionary = sentences.get_dictionary().as_dict()
empty_row = [0] * len(word_dictionary)
matrix_array = [empty_row] * sentences.size()
matrix_array = np.array(matrix_array)

for word_index, word in word_dictionary.iteritems():
    for sentence_index in range(0, sentences.size()):
        matrix_array[sentence_index, word_index] = sentences.count_word_in_sentence(sentence_index, word)

first_row = matrix_array[0,]
distance_dict = {}
sentence_index = 0
for row in matrix_array:
    distance_dict[sentence_index] = cosine(first_row, row)
    sentence_index += 1


def find_index_by_value(dict, value):
    for index, dict_value in dict.iteritems():
        if value == dict_value:
            return index


distance_values = sorted(distance_dict.values())
print distance_dict
print distance_values
first_hit = find_index_by_value(distance_dict, distance_values[1])
second_hit = find_index_by_value(distance_dict, distance_values[2])

result_file = open('result.txt', 'w')
result_file.write(str(first_hit) + ' ' + str(second_hit))
result_file.close()
