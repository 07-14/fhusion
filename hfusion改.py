# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 10:37:22 2017

@author: soujanyaporia
"""



import numpy as np
# 打印全部显示
np.set_printoptions(threshold=np.inf)#threshold='nan'出错
np.random.seed(1337) # for reproducibility

from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input,Dense,GRU,LSTM,Concatenate,Dropout,Activation,Add, Masking
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Reshape
from keras.backend import shape
from keras.utils import plot_model
from keras.layers.merge import Multiply,Concatenate
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from keras.optimizers import RMSprop,Adadelta,Adam

from keras.callbacks import Callback

import sys

# 自定义数据
def createOneHot(train_label,  test_label):


  maxlen = int(max(train_label.max(), test_label.max()))
  
  train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))
  test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
  
  for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
      train[i,j,train_label[i,j]]=1

  for i in range(test_label.shape[0]):
    for j in range(test_label.shape[1]):
      test[i,j,test_label[i,j]]=1

  return train,  test

# 得到训练效果
def calc_test_result(result, test_label, test_mask):

  true_label=[]
  predicted_label=[]

  for i in range(result.shape[0]):
    for j in range(result.shape[1]):
      if test_mask[i,j]==1:
        true_label.append(np.argmax(test_label[i,j] ))
        predicted_label.append(np.argmax(result[i,j] ))
    
  print("Confusion Matrix :")
  print(confusion_matrix(true_label, predicted_label))
  print("Classification Report :")
  print(classification_report(true_label, predicted_label,digits=4))
  print("Accuracy ", accuracy_score(true_label, predicted_label))
  print("Macro Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='macro'))
  print("Weighted Classification Report :")
  print(precision_recall_fscore_support(true_label, predicted_label,average='weighted'))
  #print "Normal Classification Report :"
  #print precision_recall_fscore_support(true_label, predicted_label)

# 分割数据
def crop(a):
    return a[:, 0:text_dim]
def crop1(a):
    return a[:, text_dim:(text_dim+visual_dim)]
def crop2(a):
    return a[:, (text_dim+visual_dim):dim]
def output_of_lambda(input_shape):
    return (input_shape[0], text_dim)
def output_of_lambda1(input_shape):
    return (input_shape[0], visual_dim)
def output_of_lambda2(input_shape):
    return (input_shape[0], audio_dim)







# ################################################################################
# #Level 2

# #concatenate level 1 output to be sent to hfusion
# fused_tensor=Concatenate(axis=2)([context_1_2,context_1_3,context_2_3])


def load_bimodal_activations():
  with open('./bimodal.pickle', 'rb') as handle:
      activations = pickle.load(handle, encoding = 'latin1')
  merged_train_data = activations['train_bimodal']
  merged_test_data = activations['test_bimodal']
  train_mask=activations['train_mask']
  test_mask=activations['test_mask']
  train_label=activations['train_label']
  test_label=activations['test_label']

  # print("In the load_bimodal_activations:merged_train_data.shape,merged_test_data.shape is ",merged_train_data.shape,merged_test_data.shape)
  # 以上输出(120, 110, 1500) (30, 110, 1500)

  #Setting dummy utterances to be 0
  for i in range(activations['train_bimodal'].shape[0]):
    for j in range(activations['train_bimodal'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(activations['test_bimodal'].shape[0]):
    for j in range(activations['test_bimodal'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  audio_dim=500
  visual_dim=500
  text_dim=500
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1] #max number of utterances per video
  dim_proj=450

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask, max_len


def load_unimodal_activations():
  # 从pickle中得到其中经过GRU训练得到的数据
  with open('./unimodal.pickle', 'rb') as handle:
      unimodal_activations = pickle.load(handle, encoding = 'latin1')

  # 数组拼接
  merged_train_data = np.concatenate((unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']), axis=2)
  merged_test_data = np.concatenate((unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']), axis=2)

  # print("In the load_unimodal_activations(): unimodal_activations['text_train'].shape,unimodal_activations['audio_train'].shape, unimodal_activations['video_train'].shape is ",unimodal_activations['text_train'].shape,unimodal_activations['audio_train'].shape, unimodal_activations['video_train'].shape)
  # 以上输出(120, 110, 100) (120, 110, 100) (120, 110, 100)
  # print("In the load_unimodal_activations:merged_train_data.shape,merged_test_data.shape is ",merged_train_data.shape,merged_test_data.shape)
  # 以上打印输出(120, 110, 300) (30, 110, 300)

  train_mask=unimodal_activations['train_mask']
  test_mask=unimodal_activations['test_mask']
  train_label=unimodal_activations['train_label']
  test_label=unimodal_activations['test_label']
  # train_mask.shape is (120, 110)
  # test_mask.shape is (30, 110)
  # train_label.shape is (120, 110, 4)
  # test_label.shape is (30, 110, 4)

  #Setting dummy utterances to be 0
  for i in range(unimodal_activations['audio_train'].shape[0]):
    for j in range(unimodal_activations['audio_train'].shape[1]):
      if train_mask[i][j] == 0.0 :

        merged_train_data[i,j,:]=0.0

  for i in range(unimodal_activations['audio_test'].shape[0]):
    for j in range(unimodal_activations['audio_test'].shape[1]):
      if test_mask[i][j] == 0.0 :

        merged_test_data[i,j,:]=0.0

  
  global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj


  audio_dim=unimodal_activations['audio_train'].shape[2]
  visual_dim=unimodal_activations['video_train'].shape[2]
  text_dim=unimodal_activations['text_train'].shape[2]
  dim = audio_dim + visual_dim + text_dim
  max_len = merged_train_data.shape[1] #max number of utterances per video
  # 原来代码dim_proj = 450
  dim_proj=400

  return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask

# 通过自定义的get_iemocap_raw函数得到iemocap数据集的训练和测试数据
def load_iemocap6way_rawdata():
    train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_iemocap_raw(
        6)

    print("train_data.shape is ", train_data.shape, "test_data.shape is ", test_data.shape, "audio_train.shape is ",
          audio_train.shape, "audio_test.shape is ", audio_test.shape, "text_train.shape is ", text_train.shape,
          "text_test.shape is ", text_test.shape, "video_train.shape is ", video_train.shape, "video_test.shape is ",
          video_test.shape, "train_label.shape is ", train_label.shape, "test_label.shape is ", test_label.shape,
          "seqlen_train.shape is ", seqlen_train.shape, "seqlen_test.shape is ", seqlen_test.shape,
          "train_mask.shape is ", train_mask.shape, "test_mask.shape is ", test_mask.shape, )
    # train_data.shape is (120, 110, 712)
    # test_data.shape is (31, 110, 712)
    # audio_train.shape is (120, 110, 100)
    # audio_test.shape is (31, 110, 100)
    # text_train.shape is (120, 110, 100)
    # text_test.shape is (31, 110, 100)
    # video_train.shape is (120, 110, 512)
    # video_test.shape is (31, 110, 512)
    # train_label.shape is (120, 110, 6)
    # test_label.shape is (31, 110, 6)
    # seqlen_train.shape is (120,)
    # seqlen_test.shape is (31,)
    # train_mask.shape is (120, 110)
    # test_mask.shape is (31, 110)

    # Setting dummy utterances to be 0
    for i in range(audio_train.shape[0]):
        for j in range(audio_train.shape[1]):
            if train_mask[i][j] == 0.0:
                train_data[i, j, :] = 0.0

    for i in range(audio_test.shape[0]):
        for j in range(audio_test.shape[1]):
            if test_mask[i][j] == 0.0:
                test_data[i, j, :] = 0.0

    global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj

    audio_dim = audio_train.shape[2]
    visual_dim = video_train.shape[2]
    text_dim = text_train.shape[2]
    dim = audio_dim + visual_dim + text_dim
    max_len = train_data.shape[1]  # max number of utterances per video
    # 原来代码dim_proj = 450
    dim_proj = 400

    return train_data, test_data, train_label, test_label, train_mask, test_mask


def load_unimodal_iemocap_6way():
    with open('./unimodal_iemocap_6way.pickle', 'rb') as handle:
        unimodal_activations = pickle.load(handle, encoding='latin1')

    # 拼接数据
    merged_train_data = np.concatenate(
        (unimodal_activations['text_train'], unimodal_activations['audio_train'], unimodal_activations['video_train']),
        axis=2)
    merged_test_data = np.concatenate(
        (unimodal_activations['text_test'], unimodal_activations['audio_test'], unimodal_activations['video_test']),
        axis=2)

    train_mask = unimodal_activations['train_mask']
    test_mask = unimodal_activations['test_mask']
    train_label = unimodal_activations['train_label']
    test_label = unimodal_activations['test_label']

    # Setting dummy utterances to be 0
    for i in range(unimodal_activations['audio_train'].shape[0]):
        for j in range(unimodal_activations['audio_train'].shape[1]):
            if train_mask[i][j] == 0.0:
                merged_train_data[i, j, :] = 0.0

    for i in range(unimodal_activations['audio_test'].shape[0]):
        for j in range(unimodal_activations['audio_test'].shape[1]):
            if test_mask[i][j] == 0.0:
                merged_test_data[i, j, :] = 0.0

    global audio_dim, visual_dim, text_dim, dim, max_len, dim_proj

    audio_dim = unimodal_activations['audio_train'].shape[2]
    visual_dim = unimodal_activations['video_train'].shape[2]
    text_dim = unimodal_activations['text_train'].shape[2]
    dim = audio_dim + visual_dim + text_dim
    max_len = merged_train_data.shape[1]  # max number of utterances per video
    # 原来代码dim_proj = 450
    dim_proj = 400

    return merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask


def Bimodal():

    # 定义论文中Fig. 2. Utterance-level bimodal fusion
  class hadamard2(Layer):
      def __init__(self, prefix, **kwargs):
          self.supports_masking = True
          self.prefix = prefix
          super(hadamard2, self).__init__(**kwargs)

      def build(self, input_shape):
          # Create a trainable weight variable for this layer.
          self.output_dim = dim_proj

          self.kernel1 = self.add_weight(name=self.prefix+'kernel1', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.kernel2 = self.add_weight(name=self.prefix+'kernel2', 
                                        shape=(self.output_dim,),
                                        initializer='TruncatedNormal',
                                        trainable=True)
          self.bias = self.add_weight(name=self.prefix+'bias', 
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
          super(hadamard2, self).build(input_shape)  # Be sure to call this somewhere!

      def call(self, x):
          assert (K.int_shape(x[0])[1] <= dim_proj)
          return K.tanh(x[0]*self.kernel1 + x[1]*self.kernel2 + self.bias)

      def compute_output_shape(self, input_shape):
          return (input_shape[0][0],self.output_dim)

  merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask = load_unimodal_activations()
  # merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask = load_iemocap6way_rawdata()
  # merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask = load_unimodal_iemocap_6way()

  #################################################################################
  #Level 1 sub-model : Hfusion

  x=Input(shape=(audio_dim+visual_dim+text_dim,), name='bi_input')
  masked = Masking(mask_value =0.0 , name='bi_mask')(x)
  x1 = Lambda(crop,output_shape=output_of_lambda, name='bi_crop1')(masked)
  x2 =Lambda(crop1,output_shape=output_of_lambda1, name='bi_crop2')(masked)
  x3 =Lambda(crop2,output_shape=output_of_lambda2, name='bi_crop3')(masked)

  # 原来代码 use_bias=False
  # dense1=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense1')(x1)
  # dense2=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense2')(x2)
  # dense3=Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense3')(x3)
  dense1 = Dense(dim_proj, activation='tanh', use_bias=True, trainable=True, name='bi_dense1')(x1)
  dense2 = Dense(dim_proj, activation='tanh', use_bias=True, trainable=True, name='bi_dense2')(x2)
  dense3 = Dense(dim_proj, activation='tanh', use_bias=True, trainable=True, name='bi_dense3')(x3)

  fused_1_2=hadamard2('1')([dense1,dense2])
  fused_1_3=hadamard2('2')([dense1,dense3])
  fused_2_3=hadamard2('3')([dense2,dense3])

  bimodal_1_2=Model(x,outputs=fused_1_2)
  bimodal_1_3=Model(x,outputs=fused_1_3)
  bimodal_2_3=Model(x,outputs=fused_2_3)


  ################################################################################
  #Level 1 sub-model : biLSTM 

  input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input1')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm1')(input_data)
  inter = Dropout(0.9, name='bi_dropout11')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis11'))(inter)
  # 原来代码output1 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis12'))(inter)
  output1 = TimeDistributed(Dense(4, activation='softmax', name='bi_tdis12'))(inter)
  lstm1 = Model(input_data, output1)
  aux1 = Model(input_data, inter)

  # 原来代码input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input2')
  input_data = Input(shape=(max_len, K.int_shape(fused_1_3)[1],), name='bi_lstm_input2')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm2')(input_data)
  inter = Dropout(0.9, name='bi_dropout21')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis21'))(inter)
  # 原来代码output2 = TimeDistributed(Dense(4, activation='softmax', name='bi_tdis22'))(inter)
  output2 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis22'))(inter)
  lstm2 = Model(input_data, output2)
  aux2 = Model(input_data, inter)

  # 原来代码 input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input3')
  input_data = Input(shape=(max_len, K.int_shape(fused_2_3)[1],), name='bi_lstm_input3')
  lstm = GRU(600, activation='tanh', return_sequences = True, dropout=0.4, name='bi_lstm3')(input_data)
  inter = Dropout(0.9, name='bi_dropout31')(lstm)
  inter = TimeDistributed(Dense(500,activation='tanh', name='bi_tdis31'))(inter)
  # 原来代码output3 = TimeDistributed(Dense(4, activation='softmax', name='bi_tdis32'))(inter)
  output3 = TimeDistributed(Dense(4,activation='softmax', name='bi_tdis32'))(inter)
  lstm3 = Model(input_data, output3)
  aux3 = Model(input_data, inter)

  ################################################################################
  #Complete Level 1 : Lstm applied on hfusion 

  main_input=Input(shape=(max_len,audio_dim+visual_dim+text_dim,), name='bi_main_input')

  video_uttr_1_2=TimeDistributed(bimodal_1_2, name='bi_tdis_b12')(main_input)
  video_uttr_1_3=TimeDistributed(bimodal_1_3, name='bi_tdis_b13')(main_input)
  video_uttr_2_3=TimeDistributed(bimodal_2_3, name='bi_tdis_b23')(main_input)
  
  aux_1_2=aux1(video_uttr_1_2)
  aux_1_3=aux2(video_uttr_1_3)
  aux_2_3=aux3(video_uttr_2_3)

  context_1_2=lstm1(video_uttr_1_2)
  BM1 = Model(main_input, context_1_2)
  Aux1 = Model(main_input, aux_1_2)
  context_1_3=lstm2(video_uttr_1_3)
  BM2 = Model(main_input, context_1_3)
  Aux2 = Model(main_input, aux_1_3)
  context_2_3=lstm3(video_uttr_2_3)
  BM3 = Model(main_input, context_2_3)
  Aux3 = Model(main_input, aux_2_3)

  #Training and fitting of Bimodals

  optimizer=Adam(lr=0.001)
  BM1.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM1.fit(merged_train_data, train_label,
                  epochs=100,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  
  train_result1 = Aux1.predict(merged_train_data)
  test_result1 = Aux1.predict(merged_test_data)

  BM2.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM2.fit(merged_train_data, train_label,
                  epochs=100,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result2 = Aux2.predict(merged_train_data)
  test_result2 = Aux2.predict(merged_test_data)

  BM3.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)
  BM3.fit(merged_train_data, train_label,
                  epochs=100,#200
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[early_stopping],
                  validation_split=0.1)
  train_result3 = Aux3.predict(merged_train_data)
  test_result3 = Aux3.predict(merged_test_data)

  train_bimodal = np.concatenate((train_result1, train_result2, train_result3), axis=2)
  test_bimodal = np.concatenate((test_result1, test_result2, test_result3), axis=2)

  bimodal_activations={}
  bimodal_activations['train_bimodal'] = train_bimodal
  bimodal_activations['test_bimodal'] = test_bimodal
  bimodal_activations['train_mask']=train_mask
  bimodal_activations['test_mask']= test_mask
  bimodal_activations['train_label']=train_label
  bimodal_activations['test_label']=test_label

  # ////////////////////////////////

  # np.set_printoptions(threshold=np.inf)

  # /////////////////输出过程数据
  print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
  print("the FT FA FV is ",merged_train_data)
  gT = Model(x, outputs=dense1)
  gA = Model(x, outputs=dense2)
  gV = Model(x, outputs=dense3)
  # ////////////////
  gT_out = TimeDistributed(gT, name='gT')(main_input)
  gA_out = TimeDistributed(gA, name='gA')(main_input)
  gV_out = TimeDistributed(gV, name='gV')(main_input)
  gT_out_pred = Model(main_input, gT_out)
  gA_out_pred = Model(main_input, gA_out)
  gV_out_pred = Model(main_input, gV_out)
  gT_data = gT_out_pred.predict(merged_train_data)
  gA_data = gA_out_pred.predict(merged_train_data)
  gV_data = gV_out_pred.predict(merged_train_data)
  gTgAgV=np.concatenate((gT_data, gA_data, gV_data), axis=2)
  print("the gT gA gV is ",gTgAgV)
  # exit(0)
  # ///////////////////////////////

  #merge all output
  # 二进制格式通常是占据磁盘空间最小、保存与加载速度最快的数据格式。最简单的方法是使用pickle。
  # pickle模块主要提供了数据持久化功能，序列化可使用dumps()函数，逆序列化使用loads()函数，将文件中的数据解析为一个python对象。
  # pickle.HIGHEST_PROTOCOL：整型，最高协议版本
  print("Saving bimodal activations")
  with open('bimodal.pickle', 'wb') as handle:
       pickle.dump(bimodal_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)


def Trimodal():

  merged_train_data, merged_test_data, train_label, test_label, train_mask, test_mask, \
              maxlen = load_bimodal_activations()

  print("In the Trimodal() merged_train_data.shape, merged_test_data.shape, train_label.shape, \
  test_label.shape, train_mask.shape, test_mask.shape,maxlen is ", merged_train_data.shape, merged_test_data.shape, train_label.shape, \
  test_label.shape, train_mask.shape, test_mask.shape,maxlen)
  # 以上输出(120, 110, 1500) (30, 110, 1500) (120, 110, 4) (30, 110, 4) (120, 110) (30, 110) 110
  #################################################################################

  class TestCallback(Callback):
      def __init__(self, test_data):
          self.test_data = test_data

      def on_epoch_end(self, epoch, logs={}):
          x, y, z = self.test_data
          result = model.predict(x)
          calc_test_result(result, y, z)
# 定义论文Fig. 3. Utterance-level trimodal hierarchical fusion
  class hadamard3(Layer):
      def __init__(self, **kwargs):
          self.supports_masking = True
          super(hadamard3, self).__init__(**kwargs)

      def build(self, input_shape):
          # Create a trainable weight variable for this layer.
          self.output_dim = 500

          self.kernel1 = self.add_weight(name='kernel1', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.kernel2 = self.add_weight(name='kernel2', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.kernel3 = self.add_weight(name='kernel3', 
                                        shape=(self.output_dim,),
                                        initializer='glorot_uniform',
                                        trainable=True)
          self.bias = self.add_weight(name='bias', 
                                        shape=(self.output_dim,),
                                        initializer='zeros',
                                        trainable=True)
          super(hadamard3, self).build(input_shape)  # Be sure to call this somewhere!

      def call(self, x, mask=None):

          def crop(a):
              return a[:, 0:500]
          def crop1(a):
              return a[:, 500:1000]
          def crop2(a):
              return a[:, 1000:1500]
          def output_of_lambda(input_shape):
              return (input_shape[0], 500)
          def output_of_lambda1(input_shape):
              return (input_shape[0], 500)
          def output_of_lambda2(input_shape):
              return (input_shape[0], 500)
          x1=Lambda(crop,output_shape=output_of_lambda)(x)
          x2=Lambda(crop1,output_shape=output_of_lambda1)(x)
          x3=Lambda(crop2,output_shape=output_of_lambda2)(x)

          ai = K.tanh(x1*self.kernel1 + x2*self.kernel2 + x3*self.kernel3 + self.bias)
          return ai

      def compute_output_shape(self, input_shape):
          return (input_shape[0],self.output_dim)
      def compute_mask(self, input, input_mask=None):
          if isinstance(input_mask, list):
              return [None] * len(input_mask)
          else:
              return None



  #################################################################################
  #Level 2: Hfusion3 trimodal
  x_tri=Input(shape=(1500,), name='tri_input')
  # 添加代码masked = Masking(mask_value=0.0, name='bi_mask')(x_tri)
  masked = Masking(mask_value=0.0, name='tri_mask')(x_tri)
  # 原来代码 fused_1_2_3=hadamard3()(x_tri)
  fused_1_2_3 = hadamard3()(masked)
  hfusion_model=Model(x_tri,outputs=fused_1_2_3)
  #################################################################################
  #Complete Level 2 : Lstm applied on hfusion3 
  main_input= Input(shape=(maxlen,1500,), name='tri_main_input')
  masked = Masking(mask_value =0.0, name='tri_masking')(main_input)
  #原来代码video_uttr = TimeDistributed(hfusion_model, name='tri_td_1')(masked)
  # video_uttr = TimeDistributed(hfusion_model, name='tri_td_1')(masked)
  video_uttr=TimeDistributed(hfusion_model, name='tri_td_1')(main_input)
  lstm = GRU(500, activation='tanh', return_sequences = True, dropout=0.35, name='tri_lstm')(video_uttr)
  lstm = Masking(mask_value=0.)(lstm)
  inter = Dropout(0.86, name="tri_dropout1")(lstm)
  # 原来代码 concatenate_=Concatenate(name="tri_concat")([video_uttr,inter])
  # 原来代码 inter1_tri = TimeDistributed(Dense(450,activation='relu', name='tri_td_2'))(concatenate_)
  inter1_tri = TimeDistributed(Dense(550, activation='relu', name='tri_td_2'))(inter)
  inter1_tri = Masking(mask_value=0.)(inter1_tri)
  context_fused_1_2_3 = Dropout(0.86, name="tri_dropout2")(inter1_tri)
  context_fused_1_2_3 = Masking(mask_value=0.)(context_fused_1_2_3)
  # 原来代码predictions = Dense(4, activation='softmax', name='tri_td_3')(context_fused_1_2_3)
  predictions = Dense(4, activation='softmax', name='tri_td_3')(context_fused_1_2_3)

  #################################################################################

  model=Model(main_input,outputs=predictions)
  optimizer=Adam(lr=0.0001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal')
                
  early_stopping = EarlyStopping(monitor='val_loss', patience=5)
  # print train_label.shape
  model.fit(merged_train_data, train_label,
                  epochs=100,
                  batch_size=20,
                  sample_weight=train_mask,
                  shuffle=True, callbacks=[TestCallback((merged_test_data,test_label,test_mask))],
                  validation_split=0.1)

  # /////////////////
  model_outData = Model(main_input, outputs=inter1_tri)
  result_outData = model_outData.predict(merged_test_data)
  print("the FTAV is ", result_outData)
  # /////////////

  result = model.predict(merged_test_data)
  calc_test_result(result, test_label, test_mask)  

# 得到iemocap数据集数据
def get_iemocap_raw(classes):
    if sys.version_info[0] == 2:
        f = open("./IEMOCAP_features_raw.pkl", "rb")
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
            f)

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
    else:
        f = open("./IEMOCAP_features_raw.pkl", "rb")
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = u.load()
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

    # print(len(trainVid))
    # print(len(testVid))

    train_audio = []
    train_text = []
    train_visual = []
    train_seq_len = []
    train_label = []

    test_audio = []
    test_text = []
    test_visual = []
    test_seq_len = []
    test_label = []
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))

    max_len = max(max(train_seq_len), max(test_seq_len))
    train_seq_len = []
    test_seq_len = []
    print('max_len', max_len)
    for vid in trainVid:
        train_seq_len.append(len(videoIDs[vid]))
        train_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        train_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        train_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        train_visual.append(video)

    for vid in testVid:
        test_seq_len.append(len(videoIDs[vid]))
        test_label.append(videoLabels[vid] + [0] * (max_len - len(videoIDs[vid])))
        pad = [np.zeros(videoText[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        text = np.stack(videoText[vid] + pad, axis=0)
        test_text.append(text)

        pad = [np.zeros(videoAudio[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        audio = np.stack(videoAudio[vid] + pad, axis=0)
        test_audio.append(audio)

        pad = [np.zeros(videoVisual[vid][0].shape)] * (max_len - len(videoIDs[vid]))
        video = np.stack(videoVisual[vid] + pad, axis=0)
        test_visual.append(video)

    train_text = np.stack(train_text, axis=0)
    train_audio = np.stack(train_audio, axis=0)
    train_visual = np.stack(train_visual, axis=0)
    # print(train_text.shape)
    # print(train_audio.shape)
    # print(train_visual.shape)

    # print()
    test_text = np.stack(test_text, axis=0)
    test_audio = np.stack(test_audio, axis=0)
    test_visual = np.stack(test_visual, axis=0)
    # print(test_text.shape)
    # print(test_audio.shape)
    # print(test_visual.shape)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    train_seq_len = np.array(train_seq_len)
    test_seq_len = np.array(test_seq_len)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(train_seq_len.shape)
    # print(test_seq_len.shape)

    train_mask = np.zeros((train_text.shape[0], train_text.shape[1]), dtype='float')
    for i in range(len(train_seq_len)):
        train_mask[i, :train_seq_len[i]] = 1.0

    test_mask = np.zeros((test_text.shape[0], test_text.shape[1]), dtype='float')
    for i in range(len(test_seq_len)):
        test_mask[i, :test_seq_len[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((train_audio, train_visual, train_text), axis=-1)
    test_data = np.concatenate((test_audio, test_visual, test_text), axis=-1)

    return train_data, test_data, train_audio, test_audio, train_text, test_text, train_visual, test_visual, train_label, test_label, train_seq_len, test_seq_len, train_mask, test_mask


def createOneHot(train_label, test_label):
    maxlen = int(max(train_label.max(), test_label.max()))

    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen + 1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen + 1))

    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i, j, train_label[i, j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i, j, test_label[i, j]] = 1

    return train, test


def test():
    x = Input(shape=(audio_dim + visual_dim + text_dim,), name='bi_input')
    masked = Masking(mask_value=0.0, name='bi_mask')(x)
    x1 = Lambda(crop, output_shape=output_of_lambda, name='bi_crop1')(masked)
    x2 = Lambda(crop1, output_shape=output_of_lambda1, name='bi_crop2')(masked)
    dense1 = Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense1')(x1)
    dense2 = Dense(dim_proj, activation='tanh', use_bias=False, trainable=True, name='bi_dense2')(x2)
    fused_1_2 = hadamard2('1')([dense1, dense2])
    bimodal_1_2 = Model(x, outputs=fused_1_2)
    input_data = Input(shape=(max_len, K.int_shape(fused_1_2)[1],), name='bi_lstm_input1')
    lstm = GRU(600, activation='tanh', return_sequences=True, dropout=0.4, name='bi_lstm1')(input_data)
    inter = Dropout(0.9, name='bi_dropout11')(lstm)
    inter = TimeDistributed(Dense(500, activation='tanh', name='bi_tdis11'))(inter)
    output1 = TimeDistributed(Dense(4, activation='softmax', name='bi_tdis12'))(inter)
    lstm1 = Model(input_data, output1)
    aux1 = Model(input_data, inter)
    main_input = Input(shape=(max_len, audio_dim + visual_dim + text_dim,), name='bi_main_input')
    video_uttr_1_2 = TimeDistributed(bimodal_1_2, name='bi_tdis_b12')(main_input)
    aux_1_2 = aux1(video_uttr_1_2)
    context_1_2 = lstm1(video_uttr_1_2)
    BM1 = Model(main_input, context_1_2)
    Aux1 = Model(main_input, aux_1_2)
def get_raw_data_iemocap(data, classes):
    videoIDs, videoSpeakers, videoLabels, videoText, videoAudio, videoVisual, videoSentence, trainVid, testVid = pickle.load(
        open("./IEMOCAP_features_raw.pkl", 'rb'), encoding='latin1')

    train_data = []
    test_data = []
    train_label = []
    test_label = []
    train_length = []
    test_length = []
    audio_train = []
    video_train = []
    text_train = []
    audio_test = []
    video_test = []
    text_test = []

    for vid in trainVid:
        text_train.append(videoText[vid])
        audio_train.append(videoAudio[vid])
        video_train.append(videoVisual[vid])
        train_label.append(videoLabels[vid])
        train_length.append(len(videoLabels[vid]))
    for vid in testVid:
        text_test.append(videoText[vid])
        audio_test.append(videoAudio[vid])
        video_test.append(videoVisual[vid])
        test_label.append(videoLabels[vid])
        test_length.append(len(videoLabels[vid]))

    text_train = keras.preprocessing.sequence.pad_sequences(text_train, maxlen=110, padding='post', dtype='float32')
    audio_train = keras.preprocessing.sequence.pad_sequences(audio_train, maxlen=110, padding='post', dtype='float32')
    video_train = keras.preprocessing.sequence.pad_sequences(video_train, maxlen=110, padding='post', dtype='float32')
    text_test = keras.preprocessing.sequence.pad_sequences(text_test, maxlen=110, padding='post', dtype='float32')
    audio_test = keras.preprocessing.sequence.pad_sequences(audio_test, maxlen=110, padding='post', dtype='float32')
    video_test = keras.preprocessing.sequence.pad_sequences(video_test, maxlen=110, padding='post', dtype='float32')

    train_label = keras.preprocessing.sequence.pad_sequences(train_label, maxlen=110, padding='post', dtype='int32')
    test_label = keras.preprocessing.sequence.pad_sequences(test_label, maxlen=110, padding='post', dtype='int32')
    # print(text_train[0, -1, :])
    # print(audio_train[0, -1, :])
    # print(video_train[0, -1, :])
    train_mask = np.zeros((text_train.shape[0], text_train.shape[1]), dtype='float')
    for i in range(len(train_length)):
        train_mask[i, :train_length[i]] = 1.0

    test_mask = np.zeros((text_test.shape[0], text_test.shape[1]), dtype='float')
    for i in range(len(test_length)):
        test_mask[i, :test_length[i]] = 1.0

    train_label, test_label = createOneHot(train_label, test_label)

    train_data = np.concatenate((audio_train, video_train, text_train), axis=-1)
    test_data = np.concatenate((audio_test, video_test, text_test), axis=-1)

    seqlen_train = train_length
    seqlen_test = test_length

    return train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask


if __name__ == '__main__':
    # output = sys.stdout
    # outputfile = open("D:\\out\\data.txt", "a")
    # type = sys.getfilesystemencoding()
    # sys.stdout = outputfile
    Bimodal()
    # Trimodal()
# 以下是将print的结果保存到硬盘文件
# output = sys.stdout
# outputfile = open("D:\\DEM4\\2.txt", "a")
# type = sys.getfilesystemencoding()  # python编码转换到系统编码输出
# sys.stdout = outputfile
# Bimodal()
# 以下是将print的结果保存到硬盘文件
# Trimodal()







