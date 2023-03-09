# 文本数据
# 目标：将文本转换成Neural Networks可以处理的东西，即数值张量；选择正确的网络结构，利用pytorch 进行nlp。
# 网络在两个级别上对文本进行操作：字符级别--依次处理一个字符；单词级别，单词是网络中最细粒度的实体。
import torch

with open('1342-0.txt',encoding='utf-8') as f:
    text = f.read()

# 解析文本中的字符，对字符进行独热编码。每个字符将由一个长度等于编码中字符数的向量表示。该向量除了有一个元素是1外其他全为0.

# 1.将文本分成若干行，任意选择一行
lines = text.split('\n')
line = lines[200]

# 2.创建一个张量，可容纳整行的独热编码的字符总数
letter_tensor = torch.zeros(len(line),128)   #128是由于ASCII的限制
# print(letter_tensor.shape)    torch.size([70,128])   78算上了空格

# 3.leeter_tensor每行将要表示一个独热编码字符。在每一行正确位置上设置1，以使每一行代表正确的字符。
# 设置1的索引对应与编码中字符的索引:
for i,letter in enumerate(line.lower().strip()):
    # 文本里含有双引号,非有效ASCII,因此在此处将其屏蔽
    letter_index = ord(letter) if ord(letter) < 128 else 0      #chr() 返回字符对应的 ASCII 数值，或者 Unicode 数值
    letter_tensor[i][letter_index] = 1
# 已经将句子独热编码成神经网络可以使用的表示形式。你也可以沿张量的行，通过建立词汇表来在词级别（word-level）对句子（即词序列）进行独热编码。\
# 由于词汇表包含许多单词，因此该方法会产生可能不是很实际的很宽的编码向量.在本章的后面将通过使用嵌入（embedding）来在单词级别表示文本

# 4.定义clean_words函数,它接受文本,将其返回小写并删除标点符号.
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list
words_in_line = clean_words(line)    #['impossible', 'mr', 'bennet', 'impossible', 'when', 'i', 'am', 'not', 'acquainted', 'with', 'him']

# 5.在编码中建立单词到索引的映射:
word_list = sorted(set(clean_words(text)))   #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
word2index_dict = {word: i for (i,word) in enumerate(word_list)}
# word2index_dict是一个字典,单词为键,整数为值.独热编码时,用该词典来有效地查找单词的索引
# print(len(word2index_dict))    7261
# print(word2index_dict['impossible'])      3394

# 6.现在专注于句子,将其分解为单词并对其进行独热编码,即每个单词使用一个独热编码向量来填充张量.创建一个空向量,赋值为句子中的单词的独热编码
word_tensor = torch.zeros(len(words_in_line),len(word2index_dict))    #words_in_line是删除标点符号.word2index_dict是删除重复了的字典
for i,word in enumerate(words_in_line):
    word_index = word2index_dict[word]
    word_tensor[i][word_index] = 1
    print('{:2}{:4}{}'.format(i,word_index,word))    #{:2}保留两个字符的位置  {:4}保留4个字符的位置.对比'{}{}{}'.format(i,word_index,word)
print(word_tensor.shape)   #torch.Size([11, 7261])  word_tensor表示长度为11编码长度为7261（这是字典中单词的数量）的一个句子









