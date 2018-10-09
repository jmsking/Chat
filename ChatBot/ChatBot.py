#! /usr/bin/python3

import tensorflow as tf
import jieba
import numpy as np
import pickle
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ChatBot():
	
	def __init__(self, enc_embed_dim = 20, dec_embed_dim = 20
				,epoches = 10, batch_size = 10, learning_rate = 0.01
				,n_enc_hidden = [20,20], n_dec_hidden = [20,20]):
		'''
		初始化方法
		Args:
			enc_embed_dim: Encoder端Embedding维度
			dec_embed_dim: Decoder端Embedding维度
			epoches: 迭代次数
			batch_size: 每批次序列个数
			learning_rate: 学习速率
			n_enc_hidden: Encoder端各隐藏层中的节点数 [N1, N2, ...]
			n_dec_hidden: Decoder端各隐藏层中的节点数 [M1, M2, ...]
		'''
		self._enc_embed_dim = enc_embed_dim
		self._dec_embed_dim = dec_embed_dim
		self._epoches = epoches
		self._batch_size = batch_size
		self._learning_rate = learning_rate
		self._n_enc_hidden = n_enc_hidden
		self._n_dec_hidden = n_dec_hidden
	
		self._source_vocab_size = 0
		self._target_vocab_size = 0
		self._source_id_word_map = None
		self._target_id_word_map = None
		self._source_word_id_map = None
		self._target_word_id_map = None
	
	def get_id_word_map(self):
		'''
		获取映射表
		'''
		return self._source_id_word_map, self._target_id_word_map, self._source_word_id_map, self._target_word_id_map
		
	def set_id_word_map(self, source_id_word_map, target_id_word_map, source_word_id_map, target_word_id_map):
		'''
		设置映射表
		'''
		self._source_id_word_map = source_id_word_map
		self._target_id_word_map = target_id_word_map
		self._source_word_id_map = source_word_id_map
		self._target_word_id_map = target_word_id_map
		
		self._source_vocab_size = len(source_id_word_map) #Encoder层输入样本字典大小
		self._target_vocab_size = len(target_id_word_map) #Decoder层输入样本字典大小
	
	def build_input_struct(self):
		'''
		构建模型输入样本结构
		Args:
			encoder_input: Encoder层Mini-batch输入
			decoder_input: Decoder层Mini-batch输入
			encoder_input_seq_len: Encoder层Mini-batch输入样本每个序列的长度
			decoder_input_seq_len: Decoder层Mini-batch输入样本每个序列的长度
			decoder_max_input_seq_len: Decoder层输入样本每批次序列最大长度
		'''
		encoder_input = tf.placeholder(dtype=tf.int32, shape=[None,None], name="encoder_input")
		decoder_input = tf.placeholder(dtype=tf.int32, shape=[None,None], name="decoder_input")
		encoder_input_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="encoder_input_seq_len")
		decoder_input_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="decoder_input_seq_len")
		decoder_max_input_seq_len = tf.reduce_max(decoder_input_seq_len, name="decoder_max_input_seq_len")
		return encoder_input, decoder_input, encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len
	
	def read_data_sets(self, path):
		'''
		读取输入文件及对应目标输出数据文件
		Args:
			path: Q&A文件路径
		Returns:
			source_ids: 对输入的索引Ids
			target_ids: 对对应目标输出的索引Ids
		'''
		source_sentences = []
		target_sentences = []
		source_words = []
		target_words = []
		# 分词处理
		is_question = True
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				if line[0] == 'M':
					line = line[2:] # 去掉'M '
					if line[-1] == '\n':
						line = line[:-1]
					word_gen = jieba.cut(line)
					if is_question:
						source_sentences.append(list(word_gen))
						is_question = False
					else:
						target_sentences.append(list(word_gen))
						is_question = True
					
		source_words = [word for sentence in source_sentences for word in sentence]
		target_words = [word for sentence in target_sentences for word in sentence]
		
		# 去重
		source_words_unique = list(set(source_words))
		target_words_unique = list(set(target_words))
		
		# 添加特殊标志
		special_words = ['<GO>', '<EOS>', '<UNK>', '<PAD>']
		
		# id-word映射
		source_id_word_map = {id:word for id, word in enumerate(source_words_unique+special_words)}
		target_id_word_map = {id:word for id, word in enumerate(target_words_unique+special_words)}
		self._source_vocab_size = len(source_id_word_map) #Encoder层输入样本字典大小
		self._target_vocab_size = len(target_id_word_map) #Decoder层输入样本字典大小
		self._source_id_word_map = source_id_word_map
		self._target_id_word_map = target_id_word_map
		
		# word-id映射
		source_word_id_map = {word:id for id, word in enumerate(source_words_unique+special_words)}
		target_word_id_map = {word:id for id, word in enumerate(target_words_unique+special_words)}
		self._source_word_id_map = source_word_id_map
		self._target_word_id_map = target_word_id_map
		
		source_sentences = source_sentences[:256]
		target_sentences = target_sentences[:256]
		# 对输入文本进行索引
		source_ids = [[source_word_id_map[word] for word in sentence] for sentence in source_sentences]
		target_ids = [[target_word_id_map[word] for word in sentence] + [target_word_id_map['<EOS>']] for sentence in target_sentences]
		
		print('source vocab size: {}'.format(self._source_vocab_size))
		print('target vocab size: {}'.format(self._target_vocab_size))
		
		# 保存字典
		pickle.dump(source_word_id_map, open('./source_word_id_map.bin','wb'))
		pickle.dump(target_word_id_map, open('./target_word_id_map.bin','wb'))
		pickle.dump(source_id_word_map, open('./source_id_word_map.bin','wb'))
		pickle.dump(target_id_word_map, open('./target_id_word_map.bin','wb'))
		
		return source_ids, target_ids
		
	def word_embedding(self, ids, vocab_size, embed_dim):
		'''
		Word Embedding
		Args:
			ids: 待Embedding的输入
		Returns:
			Embedding的序列
		'''
		return tf.contrib.layers.embed_sequence(ids, vocab_size, embed_dim)
		
	def build_encoder_layer(self):
		'''
		构建Encoder层
		Return:
			encoder_model: Encoder模型
		'''
		def build_lstm_cell(rnn_size):
			'''
			构建LSTM单元
			'''
			init = tf.random_uniform_initializer(-1, 0.2, seed = 100, dtype = tf.float32)
			return tf.contrib.rnn.LSTMCell(rnn_size, initializer = init)
		encoder_model = tf.contrib.rnn.MultiRNNCell([build_lstm_cell(rnn_size) for rnn_size in self._n_enc_hidden])	
		return encoder_model
		
	def obtain_encoder_result(self, encoder_model, embed_input):
		'''
		输入embedding Mini-batch样本
		获取Encoder模型结果
		Args:
			encoder_model: Encoder模型
			embed_input: embedding Mini-batch样本 [batch_size, seq_len, embed_dim]
		Return:
			outputs: Encoder模型RNN单元的输出 [batch_size, seq_len, rnn_size]
			last_states: Encoder模型RNN单元最后状态的输出, 一般为元组(c, h) [batch_size, rnn_size]
		'''
		outputs, last_states = tf.nn.dynamic_rnn(encoder_model, embed_input, sequence_length = self._encoder_input_seq_len, dtype=tf.float32)
		return outputs, last_states
	
	def obtain_decoder_input(self):
		'''
		获取Decoder模型的输入样本
		Return:
			decoder_input: 预处理后的Decoder输入
		'''
		#删除<EOS>标志,并添加<GO>作为Decoder输入
		#slice_input = tf.slice(self._decoder_input, [0,0], [self._batch_size, -1])
		slice_input = self._decoder_input[:,:-1]
		decoder_input = tf.concat([tf.fill([self._batch_size,1], self._target_word_id_map['<GO>']), slice_input], 1)
		return decoder_input
	
	def build_decoder_layer(self):
		'''
		构建Decoder模型
		Return:
			decoder_model: Decoder模型
		'''
		def build_lstm_cell(rnn_size):
			'''
			构建LSTM单元
			'''
			init = tf.random_uniform_initializer(-1, 0.2, seed = 100, dtype = tf.float32)
			return tf.contrib.rnn.LSTMCell(rnn_size, initializer = init)
		decoder_model = tf.contrib.rnn.MultiRNNCell([build_lstm_cell(rnn_size) for rnn_size in self._n_dec_hidden])
		return decoder_model
		
	def obtain_decoder_result(self, encoder_state, decoder_model, decoder_input):
		'''
		得到Decoder模型的输出
		Args:
			encoder_state: Encoder层状态输出
			decoder_model: 构建的Decoder模型
			decoder_input: Decoder层Mini-batch输入
		Return:
			Decoder层训练及预测结果
		'''
		decoder_embedding = tf.Variable(tf.random_uniform([self._target_vocab_size, self._dec_embed_dim]))
		embedding_decoder_input = tf.nn.embedding_lookup(decoder_embedding, decoder_input)
		
		# Decoder端输出全连接层
		output_layer = tf.layers.Dense(self._target_vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
		
		with tf.variable_scope('decode'):
			# only read inputs
			train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = embedding_decoder_input, sequence_length = self._decoder_input_seq_len, time_major=False)
			train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_model, train_helper, encoder_state, output_layer)
			train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, 
																	maximum_iterations=self._decoder_max_input_seq_len)
		with tf.variable_scope('decode', reuse = True):
			start_tokens = tf.tile(tf.constant([self._target_word_id_map['<GO>']], dtype=tf.int32), [self._batch_size])
			infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = decoder_embedding, 
																start_tokens = start_tokens,
																end_token = self._target_word_id_map['<EOS>'])
			infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_model, infer_helper, encoder_state, output_layer)
			infer_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, impute_finished=True, 
														maximum_iterations=self._decoder_max_input_seq_len)
		return train_decoder_output, infer_decoder_output
	
	
	def build_seq2seq_model(self):
		'''
		连接Encoder及Decoder形成Seq2Seq模型
		Returns:
			train_decoder_output: Decoder训练结果
			infer_decoder_output: Decoder预测结果	
		'''
		# 得到Encoder层Embedding后的Mini-batch输入
		encoder_embed_input = self.word_embedding(self._encoder_input, self._source_vocab_size, self._enc_embed_dim)
		# 构建Encoder模型
		encoder_model = self.build_encoder_layer()
		# 获取Encoder状态变量
		_, encoder_state = self.obtain_encoder_result(encoder_model, encoder_embed_input)
		# 获取Decoder输入
		decoder_input = self.obtain_decoder_input()
		# 构建Decoder模型
		decoder_model = self.build_decoder_layer()
		# 得到Decoder输出
		train_decoder_output, infer_decoder_output = self.obtain_decoder_result(encoder_state, decoder_model, decoder_input)
		return train_decoder_output, infer_decoder_output
	
	def pad_input(self, input, pad_int):
		'''
		补全输入样本
		使得每批次中各个序列的长度相等
		Args:
			input: Mini-batch样本
			pad_int: 补全符(整型)
		Return:
			padding_input: 补全之后的样本
		'''
		max_len = max([len(item) for item in input])
		padding_input = np.array([item + [pad_int]*(max_len-len(item)) for item in input])
		return padding_input
	
	
	def obtain_mini_batch(self, source_vocab_idx, target_vocab_idx):
		'''
		获取Mini-batch输入样本
		Args:
			source_vocab_idx: 输入样本索引列表
			target_vocab_idx: 理论输出索引列表
		Return:
			pad_source_input_batch: 生成的输入Mini-batch(通过<PAD>进行了对齐)
			pad_target_input_batch: 对应的理论输出Mini-batch(通过<PAD>进行了对齐)
			source_seq_len: 输入Mini-batch中每个序列的长度
			target_seq_len: 理论输出Mini-batch中每个序列的长度
		'''
		batches = len(source_vocab_idx) // self._batch_size
		for bat in range(batches):
			start = bat*self._batch_size
			source_input_batch = source_vocab_idx[start:start+self._batch_size]
			target_input_batch = target_vocab_idx[start:start+self._batch_size]
		
			pad_source_input_batch = self.pad_input(source_input_batch, self._source_word_id_map['<PAD>'])
			pad_target_input_batch = self.pad_input(target_input_batch, self._target_word_id_map['<PAD>'])
			
			source_seq_len = []
			target_seq_len = []	
			for source in source_input_batch:
				source_seq_len.append(len(source))
			for target in target_input_batch:
				target_seq_len.append(len(target))
				
			yield pad_source_input_batch, pad_target_input_batch, source_seq_len, target_seq_len
			
	def build_train_graph(self):
		'''
		构建训练图模型
		Returns:
			train_graph: 训练图
			train_op: 训练动作
			loss: 训练损失
		'''
		train_graph = tf.Graph()
		with train_graph.as_default():
			self._encoder_input, self._decoder_input \
			,self._encoder_input_seq_len, self._decoder_input_seq_len, self._decoder_max_input_seq_len = self.build_input_struct()
			train_decoder_output, infer_decoder_output = self.build_seq2seq_model()
			training_logits = tf.identity(train_decoder_output.rnn_output, 'logits')
			infer_logits = tf.identity(infer_decoder_output.sample_id, 'infer')
			masks = tf.sequence_mask(self._decoder_input_seq_len, self._decoder_max_input_seq_len, dtype=tf.float32, name='masks')
			with tf.name_scope('optimization'):
				loss = tf.contrib.seq2seq.sequence_loss(training_logits, self._decoder_input, masks)
				optimizer = tf.train.AdamOptimizer(self._learning_rate)
				gradients = optimizer.compute_gradients(loss)
				capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
				train_op = optimizer.apply_gradients(capped_gradients)
		return train_graph, train_op, loss		
			
			
	def train(self, source_vocab_idx, target_vocab_idx):
		'''
		开始训练
		Args:
			source_vocab_idx: 同read_data_sets中source_ids
			target_vocab_idx: 同read_data_sets中target_ids
		'''
		# 将样本分为训练集及验证集
		train_source_vocab_idx = source_vocab_idx[self._batch_size:]
		train_target_vocab_idx = target_vocab_idx[self._batch_size:]
		# 其中一个batch_size作为验证集
		#valid_source_vocab_idx = source_vocab_idx[:self._batch_size]
		#valid_target_vocab_idx = target_vocab_idx[:self._batch_size]
		train_graph, train_op, loss = self.build_train_graph()
		checkpoint = "./trained_model.ckpt"
		with tf.Session(graph=train_graph) as sess:
			sess.run(tf.global_variables_initializer())
			for epo in range(self._epoches):
				for bat,(pad_source_input_batch, pad_target_input_batch, source_seq_len, target_seq_len) in enumerate(		
					self.obtain_mini_batch(train_source_vocab_idx, train_target_vocab_idx)):
					_, cost = sess.run(
						[train_op, loss],
						feed_dict={
						self._encoder_input: pad_source_input_batch,
						self._decoder_input: pad_target_input_batch,
						self._encoder_input_seq_len: source_seq_len,
						self._decoder_input_seq_len: target_seq_len})
					if bat % self._batch_size == 0:
						print('Epoch: {:>3}/{} - Batch: {:>4}/{} - Training loss: {:>6.3f}'
							.format(epo+1, self._epoches, bat+1, len(train_source_vocab_idx)//self._batch_size, cost))
			# 保存模型
			saver = tf.train.Saver()
			saver.save(sess, checkpoint)
			print("Save Model Success.")

	def build_infer_input(self, input):
		'''
		构建预测时的输入样本
		Args:
			input: 原始输入
		Return:
			infer_input_seq: 处理后的输入序列
			words_length: 分词后的长度
		'''
		max_infer_seq_length = 15
		words = list(jieba.cut(input))
		words_length = len(words)
		pad = self._source_word_id_map['<PAD>']
		infer_input_seq = [self._source_word_id_map.get(item, self._source_word_id_map['<UNK>']) for item in words] + [self._source_word_id_map['<PAD>']]*(max_infer_seq_length-words_length)
		return infer_input_seq, words_length
		
	def infer(self, question):
		'''
		开始预测
		Args:
			question: 问题
		Returns:
			answer: 针对某个问题的回答
		'''
		if self._source_id_word_map is None or self._target_id_word_map is None or self._source_word_id_map is None or self._target_word_id_map is None:
			source_id_word_map = pickle.load(open('./source_id_word_map.bin','rb'))
			target_id_word_map = pickle.load(open('./target_id_word_map.bin','rb'))
			source_word_id_map = pickle.load(open('./source_word_id_map.bin','rb'))
			target_word_id_map = pickle.load(open('./target_word_id_map.bin','rb'))
			self.set_id_word_map(source_id_word_map, target_id_word_map, source_word_id_map, target_word_id_map)
		infer_input_seq, words_length = self.build_infer_input(question)

		checkpoint = "./trained_model.ckpt"

		loaded_graph = tf.Graph()
		with tf.Session(graph=loaded_graph) as sess:
			# 加载模型
			loader = tf.train.import_meta_graph(checkpoint + '.meta')
			loader.restore(sess, checkpoint)

			encoder_input = loaded_graph.get_tensor_by_name('encoder_input:0')
			logits = loaded_graph.get_tensor_by_name('infer:0')
			encoder_input_seq_len = loaded_graph.get_tensor_by_name('encoder_input_seq_len:0')
			decoder_input_seq_len = loaded_graph.get_tensor_by_name('decoder_input_seq_len:0')
			
			infer_logits = sess.run(logits, {encoder_input: [infer_input_seq]*self._batch_size, 
											encoder_input_seq_len: [words_length]*self._batch_size, 
											decoder_input_seq_len: [words_length]*self._batch_size})[0]
		pad = self._source_word_id_map["<PAD>"]
		eos = self._source_word_id_map["<EOS>"]
		answer = " ".join([self._target_id_word_map[i] for i in infer_logits if i != pad or i != eos])
		print('Origin input:', question)
		print('\nSource')
		print('	Word Number:		{}'.format([i for i in infer_input_seq]))
		print('	Input Words: {}'.format(" ".join([self._source_id_word_map[i] for i in infer_input_seq])))
		print('\nTarget')
		print('	Word Number:			 {}'.format([i for i in infer_logits if i != pad]))
		print('	Response Words: {}'.format(answer))
		return answer



if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Params should equal 2. Example: python ChatBot.py <train/predict>')
	if sys.argv[1] == 'train':
		path = 'data/xiaohuangji50w_nofenci.conv'
		chatBot = ChatBot()
		source_vocab_idx, target_vocab_idx = chatBot.read_data_sets(path)
		print('----------start training---------')
		chatBot.train(source_vocab_idx, target_vocab_idx)
		print('----------training finish--------')
	if sys.argv[1] == 'predict':
		print('----------start predict----------')
		chatBot = ChatBot()
		print('----------load dict success-----')
		answer = chatBot.infer('你在干嘛')