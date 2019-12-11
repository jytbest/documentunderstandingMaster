import pandas as pd
import  numpy as np
import yaml
from keras.layers import Input, Masking, Bidirectional, GRU, Embedding, Dense, TimeDistributed, concatenate, Conv1D, \
    Reshape, Conv2D, MaxPool2D, GlobalMaxPool2D, ZeroPadding1D, Dropout, LSTM
from keras.models import Model,model_from_yaml,load_model
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint

import keras
from keras_contrib.layers import CRF

from keras_self_attention import SeqSelfAttention



#全局变量
yaml_file = 'KTBG-attention-1211'
max_par_num = 200
num_class = 23 #分类数

#连续特征
continuous_feature_num= 10
#离散特征
#大纲级别
max_outlineLevel = 5
weight_outlineLevel = np.identity(max_outlineLevel)
#Word对象类型
max_wordobjectType = 7
weight_wordobjectType = np.identity(max_wordobjectType)
#标记类型
max_imarkersType = 10
weight_imarkersType = np.identity(max_imarkersType)
#关键字
max_keyWord = 12
weight_keyWord  = np.identity(max_keyWord)
#粗体
max_bold = 4
weight_bold = np.identity(max_bold)
#斜体
max_italic = 4
weight_italic = np.identity(max_italic)

#标记位置
max_imarkerspos = 4
weight_imarkerspos = np.identity(max_imarkerspos)

def toArray(dr, name):
    #取到每一个间隔符对应的索引位置，为了区分每篇文档
    spacer_index = dr[dr[name] == '-'].index.values
    #先把第一个文档数据处理好，第二个文档到之后可以循环实现。
    trainData = dr.iloc[0:spacer_index[0]]#选取0-索引位置的dr对应的列数据
    trainData = trainData.astype(float)  #强制类型转换
    #df.iloc 几行几列 df.iloc[1] dr对应一行n列
    b = np.zeros(shape=(max_par_num-spacer_index[0],len(dr.iloc[1])))
    trainData = np.row_stack((trainData,b))
    #len(spacer_index) 文档的篇数，遍历每篇文档
    #从第二篇文档开始到最后
    for x in range(len(spacer_index)):
        #因为横线多一行！！！
        if(x+1<len(spacer_index)):
            #与第一个文档的处理过程及其相似
            data = dr.iloc[spacer_index[x]+1:spacer_index[x+1]].values
            data = data.astype(float)
            b = np.zeros(shape=(max_par_num-(spacer_index[x+1]-(spacer_index[x]+1)),len(dr.iloc[1])))
            data = np.row_stack((data,b))
            #一篇文档一篇文档处理并 连接数据
            trainData = np.concatenate((trainData,data),axis=0)
    #处理并连接完数据
    #当传入的数据是等号这个由全部连续值组成的数据时，连续的特征是所有列数组拼成了一个矩阵
    if name == 'fontsize':
        trainData = np.reshape(trainData,newshape=(len(spacer_index),max_par_num,len(dr.iloc[1])))
    else :
        #因为离散的特征是每一个是一个数组
        trainData = np.reshape(trainData,newshape=(len(spacer_index),max_par_num))
    print(name,trainData.shape)
    return trainData

def get_keras_data(path, step):
    # 读取语料
    dr_train = pd.read_csv(path, encoding='utf-8')
    # 语料以键值对形式保存。离散型和连续型是不一样的
    #大纲级别
    outlinelevel_value = dr_train[['outlinelevel']]  # 注意是两个中括号
    #word对象类型
    wordobjecttype_value = dr_train[['wordobjecttype']]
    #标记类型
    imarkerstype_value = dr_train[['imarkerstype']]
    #关键字
    keyword_value = dr_train[['keyword']]
    #粗体
    bold_value = dr_train[['bold']]
    #斜体
    italic_value = dr_train[['italic']]
    #编号位置
    imarkerspos_value = dr_train[['imarkerspos']]
    #颜色
    fontcolor_value = dr_train[['fontcolor']]

    #连续的10个特征
    ##fontsize、characount、sentencount、linespacing、firstlineindent、spacebefore、spaceafter、wordobjectwidth、wordobjectheight
#IMarkersPos、Punctuation
    value_continuous_feature = dr_train[['fontsize', 'characount', 'sentencount', 'linespacing', 'firstlineindent', 'spacebefore', 'spaceafter', 'wordobjectwidth','wordobjectheight','punctuation']]  # 连续的

    #再赋键名
    #->调toArray()自定义函数
    #特征对应的所有值处理之后
    continuous = toArray(value_continuous_feature,'fontsize')

    outlinelevel = toArray(outlinelevel_value,'outlinelevel')
    wordobjecttype = toArray(wordobjecttype_value, 'wordobjecttype')
    imarkerstype = toArray(imarkerstype_value, 'imarkerstype')
    keyword = toArray(keyword_value, 'keyword')
    bold = toArray(bold_value, 'bold')
    italic = toArray(italic_value, 'italic')
    imarkerspos = toArray(imarkerspos_value, 'imarkerspos')
    #fontcolor = toArray(fontcolor_value, 'fontcolor')
    #给每一类特征 字典形式
    X = {
        'continuous':continuous,
        'outlinelevel':outlinelevel,
        'wordobjecttype':wordobjecttype,
        'imarkerstype':imarkerstype,
        'keyword':keyword,
        'bold':bold,
        'italic':italic,
        'imarkerspos':imarkerspos,
        #'fontcolor':fontcolor
    }
    if step!='predict':
        label_value = dr_train[['label']]
        #同样调用toarray()函数，这里是标注值，而不是特征了，相当于文本对应的标注
        label_original = toArray(label_value,'label')
        #这里调用np.utils.to_categorical实现one-hot分类。
        label = np_utils.to_categorical(label_original,num_class)
        #返回的是整个特征数据集，转化成one-hot值的段落角色，原始的段落角色
        print(label.shape)
        #print(label)
        return X,label,label_original
    else:
        return X

def GRU_model():
    input_continuous_feature = Input((max_par_num, continuous_feature_num), name='continuous')
    # 离散特征，每一个特征都分开，只有一列
    # 大纲级别
    input_outlinelevel = Input((max_par_num,), name='outlinelevel')
    emb_outlinelevel = Embedding(max_outlineLevel, max_outlineLevel, weights=[weight_outlineLevel])(
        input_outlinelevel)
    # word对象类型
    input_wordobjecttype = Input((max_par_num,), name='wordobjecttype')
    emb_wordobjecttype = Embedding(max_wordobjectType, max_wordobjectType, weights=[weight_wordobjectType])(
        input_wordobjecttype)
    # 标记类型
    input_imarkerstype = Input((max_par_num,), name='imarkerstype')
    emb_imarkerstype = Embedding(max_imarkersType, max_imarkersType, weights=[weight_imarkersType])(input_imarkerstype)
    # 关键字
    input_keyword = Input((max_par_num,), name='keyword')
    emb_keyword = Embedding(max_keyWord, max_keyWord, weights=[weight_keyWord])(input_keyword)
    # 粗体
    input_bold = Input((max_par_num,), name='bold')
    emb_bold = Embedding(max_bold, max_bold, weights=[weight_bold])(input_bold)
    # 斜体
    input_italic = Input((max_par_num,), name='italic')
    emb_italic = Embedding(max_italic, max_italic, weights=[weight_italic])(input_italic)
    # 标记位置
    input_imarkerspos = Input((max_par_num,), name='imarkerspos')
    emb_imarkerspos = Embedding(max_imarkerspos, max_imarkerspos, weights=[weight_imarkerspos])(input_imarkerspos)

    # # 颜色
    # input_fontcolor = Input((max_par_num,), name='fontcolor')
    # emb_fontcolor = Embedding(max_fontcolor, max_fontcolor, weights=[weight_fontcolor])(
    #     input_fontcolor)

    # print(input_outlineLevel)
    # 最后连接所有这些向量
    input_all = concatenate(
        [input_continuous_feature, emb_outlinelevel, emb_wordobjecttype, emb_imarkerstype, emb_keyword, emb_bold,
         emb_italic, emb_imarkerspos])

    # print(input_all.shape)#变长序列处理，用Masking层
    # input_middle = Masking(mask_value=0)(input_all)

    # att = SeqSelfAttention(
    #     attention_width=15,
    #     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    #     attention_activation=None,
    #     kernel_regularizer=keras.regularizers.l2(1e-6),
    #     use_attention_bias=False,
    #     name='Attention',
    # )(input_middle)

    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2), name='Birectional_gru_1')(input_all)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2), name='Birectional_gru_2')(x)
    # x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(x)

    att = SeqSelfAttention(
        attention_width=15,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-6),
        use_attention_bias=False,
        name='Attention',
    )(x)

    half_window_size = 2
    paddinglayer = ZeroPadding1D(padding=half_window_size)(att)

    conv = Conv1D(nb_filter=256, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
    conv_d = Dropout(0.2)(conv)

    x = TimeDistributed(Dense(num_class))(conv_d)

    crf = CRF(num_class, sparse_target=False)
    crf_output = crf(x)

    # model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model = Model(
        inputs=[input_continuous_feature, input_outlinelevel, input_wordobjecttype, input_imarkerstype, input_keyword,
                input_bold, input_italic, input_imarkerspos], output=crf_output)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss=crf.loss_function, optimizer='Nadam', metrics=[crf.accuracy])
    print(model.summary())
    return model



def predict(path):
    #test_label原始的结果拿来计算
    test_data= get_keras_data(path, 'predict')
    # 打开已经训练好的文件夹，
    # with open('./modelFile/{}.yml'.format(yaml_file),'r') as m:
    #     #加载训练好的模型
    #     yaml_string = yaml.load(m)
    # model = model_from_yaml(yaml_string)
    # 加载权重
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


    label_dic = {0: '未知', 1: '题目', 2: '作者', 3: '一级标题', 4: '文本段', 5: '参考文献标题', 6: '参考文献条目', 7: '称呼', 8: '落款人或组织', 9: '落款日期',
               10: '表格',11: '表题内容', 12: '副题目', 13: '公式', 14: '结束语', 15: '二级标题', 16: '三级标题',17:'图片',18:'四级标题',19:'图题',20:'图片说明',21:'图片',22:'五级标题'}


    y_predict = y_predict.astype(int)

    predict_list = []  #保存预测标签结果

    for x in y_predict:
        for i in x:
            predict_list.append((label_dic[i]))
    output_compare = {"预测":predict_list}

    dataFrame = pd.DataFrame(output_compare)
    dataFrame.to_csv("output/compareResult1211-prediction.csv",index = False,encoding ='utf-8')



if __name__ == '__main__':
    predict('./input/KTBGprediction.csv')
