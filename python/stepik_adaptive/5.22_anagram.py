import collections

word1 = input().lower()
word2 = input().lower()

print(collections.Counter(word2) == collections.Counter(word1))