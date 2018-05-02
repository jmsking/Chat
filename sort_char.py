#! /usr/bin/python3

import tensorflow as tf
import numpy as np

# ------- 超参数设置 --------
# 迭代次数
epoches = 10
# 每批次序列个数
batch_size = 128
# RNN模型LSTM单元个数
rnn_size = 30
# RNN模型隐藏层层数
rnn_layer = 2
# Embedding维度
encoder_embed_dim = 20
decoder_embed_dim = 20
# 学习速率
learning_rate = 0.01
# 每隔ITER次打印一次
ITER = 1000


def build_input(input_path, is_target):
  '''
  构造输入样本
  Params:
    input_path: 训练样本路径
    is_target: 处理的输入文本是否是对应目标输出文件
  Return:
    vocab_idx: 输入样本的索引列表
  len(ch_to_int: 样本序列的长度
  ch_to_int: 输入样本字符-索引映射
  int_to_ch: 输入样本索引-字符映射
  '''
  with open(input_path, 'r') as f:
    text = f.read()
    #print(text)
  words = text.split('\n')
  #print(words)
  special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
  vocab = list(set([ch for item in (words) for ch in item])) #去重
  #print(vocab)
  int_to_ch = {idx : ch for idx, ch in enumerate(vocab+special_words)}
  ch_to_int = {ch : idx for idx, ch in enumerate(vocab+special_words)}
  # 将各个单词用索引进行表示
  if is_target:
    vocab_idx = [[ch_to_int.get(ch, ch_to_int['<UNK>']) for ch in item] + [ch_to_int['<EOS>']] for item in words]
  else:
    vocab_idx = [[ch_to_int.get(ch, ch_to_int['<UNK>']) for ch in item] for item in words]
  return vocab_idx, len(ch_to_int), ch_to_int, int_to_ch

def build_input_struct():
  '''
  构建模型输入样本结构
  '''
  encoder_input = tf.placeholder(dtype=tf.int32, shape=[None,None], name="encoder_input")
  decoder_input = tf.placeholder(dtype=tf.int32, shape=[None,None], name="decoder_input")
  encoder_input_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="encoder_input_seq_len")
  decoder_input_seq_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="decoder_input_seq_len")
  decoder_max_input_seq_len = tf.reduce_max(decoder_input_seq_len, name="decoder_max_input_seq_len")
  #print(decoder_max_input_seq_len)
  return encoder_input, decoder_input, encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len

def embedding_encoder_input(input, vocab_size, embed_dim):
  '''
  对encoder输入进行embedding
  Params:
    input: 输入样本
    vocab_size: 字典单词个数
    embed_dim: embedding维数
  Return:
    embedding处理后的样本
  '''
  return tf.contrib.layers.embed_sequence(input, vocab_size, embed_dim)

def build_encoder_layer(rnn_size, rnn_layer):
  '''
  构建Encoder层
  Params:
    rnn_size: 每层RNN(LSTM)单元个数
    rnn_layer: Encoder层数
  Return:
    encoder_model: Encoder模型
  '''
  def build_lstm_cell(rnn_size):
    '''
    构建LSTM单元
    '''
    init = tf.random_uniform_initializer(-1, 0.2, seed = 100, dtype = tf.float32)
    return tf.contrib.rnn.LSTMCell(rnn_size, initializer = init)
  encoder_model = tf.contrib.rnn.MultiRNNCell([build_lstm_cell(rnn_size) for _ in range(rnn_layer)])  
  return encoder_model
  
def obtain_encoder_result(encoder_model, embed_input, input_seq_len):
  '''
  输入embedding Mini-batch样本
  获取Encoder模型结果
  Params:
    encoder_model: Encoder模型
    embed_input: embedding Mini-batch样本 [batch_size, seq_len, embed_dim]
    input_seq_len: Mini-batch样本序列长度 [batch_size] 
  Return:
    outputs: Encoder模型RNN单元的输出 [batch_size, seq_len, rnn_size]
    last_states: Encoder模型RNN单元最后状态的输出, 一般为元组(c, h) [batch_size, rnn_size]
  '''
  outputs, last_states = tf.nn.dynamic_rnn(encoder_model, embed_input, sequence_length = input_seq_len, dtype=tf.float32)
  return outputs, last_states
  
def obtain_decoder_input(target_input, batch_size):
  '''
  获取Decoder模型的输入样本
  Params:
    target_inputs: Decoder理论输出样本
    batch_size: 序列个数
  Return:
    decoder_input: 预处理后的Decoder输入
  '''
  #删除<EOS>标志,并添加<GO>作为Decoder输入
  slice_input = tf.slice(target_input, [0,0], [batch_size,-1])
  decoder_input = tf.concat([tf.fill([batch_size,1], target_ch_to_int['<GO>']), slice_input], 1)
  return decoder_input
  
def build_decoder_layer(rnn_size, rnn_layer):
  '''
  构建Decoder模型
  Params:
    rnn_size: 每层RNN(LSTM)单元个数
    rnn_layer: Decoder层数
  Return:
    decoder_model: Decoder模型
  '''
  def build_lstm_cell(rnn_size):
    '''
    构建LSTM单元
    '''
    init = tf.random_uniform_initializer(-1, 0.2, seed = 100, dtype = tf.float32)
    return tf.contrib.rnn.LSTMCell(rnn_size, initializer = init)
  decoder_model = tf.contrib.rnn.MultiRNNCell([build_lstm_cell(rnn_size) for _ in range(rnn_layer)])  
  return decoder_model
  
def obtain_decoder_result(encoder_state, decoder_model, 
            decoder_input, embed_dim, vocab_size, input_seq_len,
            decoder_max_input_seq_len):
  '''
  得到Decoder模型的输出
  Params:
    encoder_state: Encoder层状态输出
    decoder_model: 构建的Decoder模型
    decoder_input: Decoder层Mini-batch输入
    embed_dim: Embedding维度
    vocab_size: Decoder层输入样本字典大小
    input_seq_len: 输入序列中每个样本的长度 [batch_size]
    decoder_max_input_seq_len: Decoder层输入样本每批次序列最大长度
  Return:
    Decoder层训练及预测结果
  '''
  decoder_embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim]))
  embedding_decoder_input = tf.nn.embedding_lookup(decoder_embedding, decoder_input)
  with tf.variable_scope('decode'):
    train_helper = tf.contrib.seq2seq.TrainingHelper(inputs = embedding_decoder_input, sequence_length = input_seq_len, time_major=False)
    train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_model, train_helper, encoder_state)
    train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, 
                                maximum_iterations=decoder_max_input_seq_len)
  with tf.variable_scope('decode', reuse = True):
    start_tokens = tf.tile(tf.constant([target_ch_to_int['<GO>']], dtype=tf.int32), [batch_size])
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = decoder_embedding, 
                              start_tokens = start_tokens,
                              end_token = target_ch_to_int['<EOS>'])
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_model, infer_helper, encoder_state)
    infer_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(infer_decoder, impute_finished=True, 
                          maximum_iterations=decoder_max_input_seq_len)
  return train_decoder_output, infer_decoder_output

def build_seq2seq_model(encoder_input, decoder_input, 
            source_vocab_size, target_vocab_size, 
            encoder_embed_dim, decoder_embed_dim, 
            encoder_input_seq_len, decoder_input_seq_len,
            decoder_max_input_seq_len):
  '''
  连接Encoder及Decoder形成Seq2Seq模型
  Params:
    encoder_input: Encoder层Mini-batch输入
    decoder_input: Decoder层Mini-batch输入
    source_vocab_size: Encoder层输入样本字典大小
    target_vocab_size: Decoder层输入样本字典大小
    encoder_embed_dim: Encoder层Embedding维度
    decoder_embed_dim: Decoder层Embedding维度
    encoder_input_seq_len: Encoder层Mini-batch输入样本每个序列的长度
    decoder_input_seq_len: Decoder层Mini-batch输入样本每个序列的长度
    decoder_max_input_seq_len: Decoder层输入样本每批次序列最大长度
  Returns:
    train_decoder_output: Decoder训练结果
    infer_decoder_output: Decoder预测结果  
  '''
  # 得到Encoder层Embedding后的Mini-batch输入
  encoder_embed_input = embedding_encoder_input(encoder_input, source_vocab_size, encoder_embed_dim)
  #print(encoder_embed_input.get_shape())
  #print(encoder_input_seq_len)
  # 构建Encoder模型
  encoder_model = build_encoder_layer(rnn_size, rnn_layer)
  # 获取Encoder状态变量
  _, encoder_state = obtain_encoder_result(encoder_model, encoder_embed_input, encoder_input_seq_len)
  # 获取Decoder输入
  decoder_input = obtain_decoder_input(decoder_input, batch_size)
  # 构建Decoder模型
  decoder_model = build_decoder_layer(rnn_size, rnn_layer)
  # 得到Decoder输出
  train_decoder_output, infer_decoder_output = obtain_decoder_result(encoder_state, decoder_model, 
            decoder_input, decoder_embed_dim, target_vocab_size, decoder_input_seq_len, decoder_max_input_seq_len)
  return train_decoder_output, infer_decoder_output
  
def pad_input(input, pad_int):
  '''
  补全输入样本
  使得每批次中各个序列的长度相等
  Params:
    input: Mini-batch样本
    pad_int: 补全符(整型)
  Return:
    pad_input: 补全之后的样本
  '''
  max_len = max([len(item) for item in input])
  pad_input = np.array([item + [pad_int]*(max_len-len(item)) for item in input])
  return pad_input
  
  
def obtain_mini_batch(source_vocab_idx, target_vocab_idx, batch_size):
  '''
  获取Mini-batch输入样本
  Params:
    source_vocab_idx: 输入样本索引列表
    target_vocab_idx: 理论输出索引列表
    batch_size: 每个Mini-batch序列个数
  Return:
    source_input_batch: 生成的输入Mini-batch
    target_input_batch: 对应的理论输出Mini-batch
    source_seq_len: 输入Mini-batch中每个序列的长度
    target_seq_len: 理论输出Mini-batch中每个序列的长度
  '''
  batches = len(source_vocab_idx) // batch_size
  for bat in range(batches):
    start = bat*batch_size
    source_input_batch = source_vocab_idx[start:start+batch_size]
    target_input_batch = target_vocab_idx[start:start+batch_size]
  
    pad_source_input_batch = pad_input(source_input_batch, source_ch_to_int['<PAD>'])
    pad_target_input_batch = pad_input(target_input_batch, target_ch_to_int['<PAD>'])
    
    source_seq_len = []
    target_seq_len = []  
    for source in source_input_batch:
      source_seq_len.append(len(source))
    for target in target_input_batch:
      target_seq_len.append(len(target))
      
    yield pad_source_input_batch, pad_target_input_batch, source_seq_len, target_seq_len
    
def build_train_graph():
  '''
  构建训练图模型
  '''
  train_graph = tf.Graph()
  with train_graph.as_default():
    encoder_input, decoder_input, encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len = build_input_struct()
    train_decoder_output, infer_decoder_output = build_seq2seq_model(encoder_input, decoder_input,
            source_vocab_size, target_vocab_size,
            encoder_embed_dim, decoder_embed_dim,
            encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len)
    training_logits = tf.identity(train_decoder_output.rnn_output, 'logits')
    infer_logits = tf.identity(infer_decoder_output.sample_id, 'infer')
    masks = tf.sequence_mask(decoder_input_seq_len, decoder_max_input_seq_len, dtype=tf.float32, name='masks')
    with tf.name_scope('optimization'):
      loss = tf.contrib.seq2seq.sequence_loss(training_logits, decoder_input, masks)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = optimizer.compute_gradients(loss)
      capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
      train_op = optimizer.apply_gradients(capped_gradients)
  return train_graph    
    
    
def train(source_vocab_idx, target_vocab_idx, batch_size):
  '''
  开始训练
  '''
  # 将样本分为训练集及验证集
  train_source_vocab_idx = source_vocab_idx[batch_size:]
  train_target_vocab_idx = target_vocab_idx[batch_size:]
  # 其中一个batch_size作为验证集
  valid_source_vocab_idx = source_vocab_idx[:batch_size]
  valid_target_vocab_idx = target_vocab_idx[:batch_size]
  
  checkpoint = "trained_model.ckpt"
  with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    for epo in range(epoches):
      for bat,(source_input_batch, target_input_batch, source_seq_len, target_seq_len) in enumerate(    
        obtain_mini_batch(train_source_vocab_idx, train_target_vocab_idx, batch_size)):
        #print(source_seq_len)
        _, loss = sess.run(
          [train_op, loss],
          feed_dict={encoder_input: source_input_batch,
          decoder_input: target_input_batch,
          encoder_input_seq_len: source_seq_len,
          decoder_input_seq_len: target_seq_len})
        if bat % print_ans == 0:
          print('Epoch: {:>3f}/{} - Training loss: {:>6.3f}'
            .format(epo, epoches, loss))
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print("模型训练及保存成功")

def build_infer_input(input):
  '''
  构建预测时的输入样本
  Params:
    input: 原始输入
  Return:
    infer_input_seq: 处理后的输入序列
  '''
  max_infer_seq_length = 7
  infer_input_seq = [source_ch_to_int.get(item, source_ch_to_int['<UNK>']) for item in input] + [source_ch_to_int['<PAD>']*(max_infer_seq_length-len(input))]
  return infer_input_seq
  
def infer():
  '''
  开始预测
  '''
  # 输入一个单词
  input = 'common'
  infer_input_seq = build_infer_input(input)


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
    
    infer_logits = sess.run(logits, {encoder_input: [infer_input_seq]*batch_size, 
                                      encoder_input_seq_len: [len(input)]*batch_size, 
                                      decoder_input_seq_len: [len(input)]*batch_size})[0]
  pad = source_ch_to_int["<PAD>"] 
  print('原始输入:', input)
  print('\nSource')
  print('  Word 编号:    {}'.format([i for i in infer_input_seq]))
  print('  Input Words: {}'.format(" ".join([source_int_to_ch[i] for i in infer_input_seq])))
  print('\nTarget')
  print('  Word 编号:       {}'.format([i for i in infer_logits if i != pad]))
  print('  Response Words: {}'.format(" ".join([target_int_to_ch[i] for i in infer_logits if i != pad])))



  
source_path = 'data/source.txt'
target_path = 'data/target.txt'

source_vocab_idx, source_vocab_size, source_ch_to_int, source_int_to_ch = build_input(source_path, False)
target_vocab_idx, target_vocab_size, target_ch_to_int, target_int_to_ch = build_input(target_path, True)

#print(source_vocab_idx)
#print(target_vocab_idx)

train_graph = tf.Graph()
with train_graph.as_default():
  encoder_input, decoder_input, encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len = build_input_struct()
  train_decoder_output, infer_decoder_output = build_seq2seq_model(encoder_input, decoder_input,
            source_vocab_size, target_vocab_size,
            encoder_embed_dim, decoder_embed_dim,
            encoder_input_seq_len, decoder_input_seq_len, decoder_max_input_seq_len)
  training_logits = tf.identity(train_decoder_output.rnn_output, 'logits')
  infer_logits = tf.identity(infer_decoder_output.sample_id, 'infer')
  masks = tf.sequence_mask(decoder_input_seq_len, decoder_max_input_seq_len, dtype=tf.float32, name='masks')
  with tf.name_scope('optimization'):
    cost = tf.contrib.seq2seq.sequence_loss(training_logits, decoder_input, masks)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

# 将样本分为训练集及验证集
train_source_vocab_idx = source_vocab_idx[batch_size:]
train_target_vocab_idx = target_vocab_idx[batch_size:]
# 其中一个batch_size作为验证集
valid_source_vocab_idx = source_vocab_idx[:batch_size]
valid_target_vocab_idx = target_vocab_idx[:batch_size]
  
checkpoint = "./trained_model.ckpt"
with tf.Session(graph=train_graph) as sess:
  sess.run(tf.global_variables_initializer())
  for epo in range(epoches):
    for bat,(source_input_batch, target_input_batch, source_seq_len, target_seq_len) in enumerate(    
      obtain_mini_batch(train_source_vocab_idx, train_target_vocab_idx, batch_size)):
      #print(source_seq_len)
      _, loss = sess.run(
        [train_op, cost],
        feed_dict={encoder_input: source_input_batch,
          decoder_input: target_input_batch,
          encoder_input_seq_len: source_seq_len,
          decoder_input_seq_len: target_seq_len})          
      if bat % ITER == 0:
        print('Epoch: {:>3}/{} - Batch: {:>4}/{} - Training loss: {:>6.3f}'
            .format(epo+1, epoches, bat, len(train_source_vocab_idx)//batch_size, loss))
  # 保存模型
  saver = tf.train.Saver()
  saver.save(sess, checkpoint)
  print("模型训练及保存成功")
  
  print("开始预测")
  infer()