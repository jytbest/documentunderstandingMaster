import pandas as pd
import numpy as np
import yaml
from keras.layers import Input, Masking, Bidirectional, GRU, Embedding, Dense, TimeDistributed, concatenate, Conv1D, \
    Reshape, Conv2D, MaxPool2D, GlobalMaxPool2D, LSTM, K, Permute, multiply, Flatten, Multiply, RepeatVector, Lambda
from keras.models import Model,model_from_yaml,load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping,ModelCheckpoint

import keras


from keras_self_attention import SeqSelfAttention

#填写❌
yaml_file = 'XWLW'
#学位论文 角色多
max_par_num = 900
#连续特征共有8个
continuous_feature_num= 8
num_class = 20


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



def toArray(dr, name):
    #取到每一个间隔符对应的索引位置，为了区分每篇文档
    spacer_index = dr[dr[name] == '-'].index.values
    #先把第一个文档数据处理好，第二个文档到之后可以循环实现。
    trainData = dr.iloc[0:spacer_index[0]]#选取0-索引位置的dr对应的列数据
    trainData = trainData.astype(float)  #强制类型转换
    #df.iloc 几行几列 df.iloc[1] dr对应一行n列
    b = np.zeros(shape=(900-spacer_index[0],len(dr.iloc[1])))

    #行组合，横着组合，但是w呢？？？？？

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
    outlineLevel_value = dr_train[['大纲级别']]
    value_continuous_feature = dr_train[['等号','字形','字数','标点','中文比例','邮箱符号','字号','缩进']]
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

        return X,label,label_original

    else:
        return X

def GRU_model():
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

    x = TimeDistributed(Dense(num_class, activation='softmax'))(att)

    #model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model = Model(inputs=[input_continuous_feature,input_wordObject,input_number,input_numberLocation,input_keyword,input_alignment,input_outlineLevel],output = x)
    # #model = keras.models.Sequential()
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['categorical_accuracy'] )
    return model

def predict(path):
    test_data= get_keras_data(path, 'predict')
    model = GRU_model()
    model.load_weights('./modelFile/{}.hdf5'.format(yaml_file))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    y_predict = []
    y_predict_pro = model.predict(test_data)
    for i in y_predict_pro:
        everyPredict = []
        for j in i:
            #取出j中元素最大值所对应的索引,将索引存入一个预测的数组中
            ##将概率最大的值的标签数值取出来，取的就是段落角色对应的数值。
            everyPredict.append(np.argmax(j))
        y_predict.append(everyPredict)

    y_predict = np.array(y_predict)
    #定义的角色标签
    label_dic = {0: '未知', 1: '题目', 2: '中文摘要', 3: '中文关键词', 4: '英文摘要', 5: '英文关键词', 6: '图片', 7: '图题', 8: '表格', 9: '表题',
                 10: '一级标题',11: '二级标题', 12: '三级标题', 13: '四级标题', 14: '五级标题', 15: '文本段', 16: '程序代码', 17: '公式', 18: '结束语标题',19: '结束语内容'}
    y_predict = y_predict.astype(int)
    predict_list = []  #保存预测标签结果

    for x in y_predict:
        for i in x:
            predict_list.append((label_dic[i]))
    output_compare = {"预测":predict_list}

    dataFrame = pd.DataFrame(output_compare)
    dataFrame.to_csv("output/compareResult-prediction.csv",index = False,encoding ='utf-8')


if __name__ == '__main__':
    predict('./input/LWprediction.csv')






