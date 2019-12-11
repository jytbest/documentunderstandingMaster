#注意力模型结合双向RNN实现对文本的分类。HN-ATT

#格式、内容和语义三个方面得到最终的识别模型。
import pandas as pd
import numpy as np
import yaml
from keras.layers import Input, Masking, Bidirectional, GRU, Embedding, Dense, TimeDistributed, concatenate, Conv1D, \
    Reshape, Conv2D, MaxPool2D, GlobalMaxPool2D, LSTM, K, Permute, multiply, Flatten, Multiply, RepeatVector, Lambda
from keras.models import Model,model_from_yaml,load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras

from keras_self_attention import SeqSelfAttention


#from keras_contrib.layers.crf import CRF
from crf_keras import CRF


from  Attention import AttentionLayer, Attention_layer, AttLayer


#填写❌
yaml_file = 'XWLW'
#学位论文 角色多
max_par_num = 900
#连续特征共有8个
continuous_feature_num= 8

#word对象
max_wordObject = 5
weight_wordObject = np.identity(max_wordObject)

#编号
max_num = 11
weight_number = np.identity(max_num)

#编号位置
max_numberLocation = 4
weight_numberLocation = np.identity(max_numberLocation)

#关键字
max_keyword = 9
weight_keyword = np.identity(max_keyword)

#对齐方式
max_alignment = 6
weight_alignment = np.identity(max_alignment)

#大纲级别
max_outlineLevel = 11
weight_outlinelevel = np.identity(max_outlineLevel)




#类别，，，，，，，，，后期会改 ❌
num_class = 20
#将每一类特征（离散每个，连续所有）拼成一个多维矩阵，且都加上了偏置b,返回矩阵值！
def toArray(dr, name):
    #取到每一个间隔符对应的索引位置，为了区分每篇文档
    spacer_index = dr[dr[name] == '-'].index.values
    #先把第一个文档数据处理好，第二个文档到之后可以循环实现。
    trainData = dr.iloc[0:spacer_index[0]]#选取0-索引位置的dr对应的列数据
    trainData = trainData.astype(float)  #强制类型转换
    #df.iloc 几行几列 df.iloc[1] dr对应一行n列
    b = np.zeros(shape=(900-spacer_index[0],len(dr.iloc[1])))

    trainData = np.row_stack((trainData,b))
    #len(spacer_index) 文档的篇数，遍历每篇文档
    #从第二篇文档开始到最后
    for x in range(len(spacer_index)):
        #因为横线多一行！！！
        if(x+1<len(spacer_index)):
            #与第一个文档的处理过程及其相似
            data = dr.iloc[spacer_index[x]+1:spacer_index[x+1]].values
            data = data.astype(float)
            b = np.zeros(shape=(900-(spacer_index[x+1]-(spacer_index[x]+1)),len(dr.iloc[1])))
            data = np.row_stack((data,b))
            #一篇文档一篇文档处理并 连接数据
            trainData = np.concatenate((trainData,data),axis=0)
    #处理并连接完数据
    #当传入的数据是等号这个由全部连续值组成的数据时，连续的特征是所有列数组拼成了一个矩阵
    if name == '等号':
        trainData = np.reshape(trainData,newshape=(len(spacer_index),max_par_num,len(dr.iloc[1])))
    else :
        #因为离散的特征是每一个是一个数组
        trainData = np.reshape(trainData,newshape=(len(spacer_index),max_par_num))
    print(name,trainData.shape)
    return trainData

#传入的参数是输入的csv语料，标注是train阶段
def get_keras_data(path, step):
    #读取语料
    dr_train = pd.read_csv(path,encoding='utf-8')
    #语料以键值对形式保存。离散型和连续型是不一样的
    #先提值
    keyword_value = dr_train[['关键字']]  #注意是两个中括号
    number_value = dr_train[['编号']]
    numberLocation_value = dr_train[['编号位置']]
    wordObject_value = dr_train[['word对象']]
    alignment_value = dr_train[['对齐方式']]
    outlineLevel_value = dr_train[['大纲级别']]#离散的
    value_continuous_feature = dr_train[['等号','字形','字数','标点','中文比例','邮箱符号','字号','缩进']]  #连续的
    #再赋键名
    #->调toArray()自定义函数
    #特征对应的所有值处理之后
    continuous = toArray(value_continuous_feature,'等号')
    wordObject = toArray(wordObject_value,'word对象')
    number = toArray(number_value,'编号')
    numberLocation =toArray(numberLocation_value,'编号位置')
    keyword = toArray(keyword_value,'关键字')
    alignment = toArray(alignment_value,'对齐方式')
    outlineLevel = toArray(outlineLevel_value,'大纲级别')
    #给每一类特征 字典形式
    X = {
        'continuous':continuous,
        'wordObject':wordObject,
        'number':number,
        'numberLocation':numberLocation,
        'keyword':keyword,
        'alignment':alignment,
        'outlineLevel':outlineLevel
    }
    if step!='predict':
        label_value = dr_train[['段落角色']]
        #同样调用toarray()函数，这里是标注值，而不是特征了，相当于文本对应的标注
        label_original = toArray(label_value,'段落角色')
        #这里调用np.utils.to_categorical实现one-hot分类。
        label = np_utils.to_categorical(label_original,num_class)
        #返回的是整个特征数据集，转化成one-hot值的段落角色，原始的段落角色

        print(label.shape)
        #print(label)
        return X,label,label_original
    else:
        return X

#返回的是双向LSTM网络
def LSTM_model():
    input_continuous_feature = Input((max_par_num,continuous_feature_num),name='continuous')
    #离散特征，每一个特征都分开，只有一列
    #word对象
    input_wordObject = Input((max_par_num,),name='wordObject')
    emb_wordObject = Embedding(max_wordObject,max_wordObject,mask_zero=True,weights=[weight_wordObject])(input_wordObject)
    # 编号
    input_number = Input((max_par_num,), name='number')
    emb_number = Embedding(max_num, max_num,  weights=[weight_number])(input_number)
    #编号位置
    input_numberLocation = Input((max_par_num,),name='numberLocation')
    emb_numberLocation = Embedding(max_numberLocation,max_numberLocation,mask_zero=True,weights=[weight_numberLocation])(input_numberLocation)
    #关键字
    input_keyword = Input((max_par_num,),name='keyword')
    emb_keyword = Embedding(max_keyword,max_keyword,mask_zero=True,weights=[weight_keyword])(input_keyword)
    #对齐方式
    input_alignment = Input((max_par_num,),name='alignment')
    emb_alignment = Embedding(max_alignment,max_alignment,mask_zero=True,weights=[weight_alignment])(input_alignment)
    #大纲级别
    input_outlineLevel = Input((max_par_num,),name='outlineLevel')
    emb_outlineLevel = Embedding(max_outlineLevel,max_outlineLevel,mask_zero=True,weights=[weight_outlinelevel])(input_outlineLevel)
    #print(input_outlineLevel)
    #最后连接所有这些向量
    input_all = concatenate([input_continuous_feature,emb_wordObject,emb_number,emb_numberLocation,emb_keyword,emb_alignment,emb_outlineLevel])



    #print(input_all.shape)#变长序列处理，用Masking层
    input_middle = Masking(mask_value=0)(input_all)

    # x = Bidirectional(GRU(128, return_sequences=True,dropout=0.2))(input_middle)
    # #将128改成了64
    # x = Bidirectional(GRU(128,return_sequences=True,dropout=0.2))(x)
    #
    #
    # # x = TimeDistributed(Dense(num_class,activation='softmax'))(x)


    # x = Bidirectional(GRU(128,return_sequences=True,dropout=0.2))(input_middle)
    #
    # x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)
    # att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    #                        kernel_regularizer=keras.regularizers.l2(1e-4),
    #                        bias_regularizer=keras.regularizers.l1(1e-4),
    #                        attention_regularizer_weight=1e-4,
    #                        attention_activation=None,
    #                        name='Attention')(x)
    # dense = TimeDistributed(Dense(num_class))(att)
    # model = Model(inputs=[input_continuous_feature,input_wordObject,input_number,input_numberLocation,input_keyword,input_alignment,input_outlineLevel], output=dense)
    # model.compile(
    #     optimizer='Nadam',
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'],
    # )




    # #
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(input_middle)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)

    att= SeqSelfAttention(
        attention_width=15,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attention',
    )(x)
    # att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    #                        kernel_regularizer=keras.regularizers.l2(1e-4),
    #                        bias_regularizer=keras.regularizers.l1(1e-4),
    #                        attention_regularizer_weight=1e-4,
    #                        name='Attention')(x)

    x = TimeDistributed(Dense(num_class, activation='softmax'))(att)

    #model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model = Model(inputs=[input_continuous_feature,input_wordObject,input_number,input_numberLocation,input_keyword,input_alignment,input_outlineLevel],output = x)
    # #model = keras.models.Sequential()
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'] )
    return model

def train():
    #首先调用get_keras_data自定义函数->
    train_data,train_label,_ = get_keras_data('./input/train.csv','train')
    #调自定义的LSTM模型！！！接收双向LSTM网络
    model = LSTM_model()
    # 创建一个权重文件保存文件夹logs
    log_dir = "logFiles/"
    # 记录所有训练过程，每隔一定步数记录最大值
    tensorboard = TensorBoard(log_dir=log_dir)
    #Checkpoint是模型的权重
    checkpoint = ModelCheckpoint("./modelFiles/{}.hdf5".format(yaml_file),monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #防止过拟合
    early = EarlyStopping(monitor="val_loss",mode="min",patience=20)
    #加了一个tensorboard
    callbacks_list = [tensorboard,checkpoint,early]
    #喂入模型值！
    model.fit(train_data,train_label,epochs=25,validation_split=0.2,batch_size=4,shuffle=False,callbacks=callbacks_list)
    #保存模型
    yaml_string = model.to_yaml()
    model.save('./modelFiles/{}.h5'.format(yaml_file))
    with open('./modelFiles/{}.yml'.format(yaml_file), 'w') as modelfile:
        modelfile.write(yaml.dump(yaml_string, default_flow_style=True))








# def attention_3d_block(inputs,single_attention_vector=False):
#     # input_dim = int(inputs.shape[2])
#
#     time_steps = K.int_shape(inputs)[1]
#     input_dim = K.int_shape(inputs)[2]
#     a = Permute((2, 1))(inputs)
#     a = Dense(time_steps, activation='softmax')(a)
#     if single_attention_vector:
#         a = Lambda(lambda x: K.mean(x, axis=1))(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((2, 1),name='attention_vec')(a)
#     output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#     return output_attention_mul
#
#
#     #
#     # a = Dense(num_class, activation='softmax')(a)
#     # a_probs = Permute((2, 1), name='attention_vec')(a)
#     # # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
#     # output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#     # return output_attention_mul



if __name__ == '__main__':
    train()
