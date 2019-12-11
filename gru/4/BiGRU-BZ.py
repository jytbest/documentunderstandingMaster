import pandas as pd
import numpy as np
import yaml
from keras.layers import Input, Masking, Bidirectional, GRU, Embedding, Dense, TimeDistributed, concatenate, Conv1D, \
    Reshape, Conv2D, MaxPool2D, GlobalMaxPool2D, LSTM, K, Permute, multiply, Flatten, Multiply, RepeatVector, Lambda
from keras.models import Model,model_from_yaml,load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras_contrib.layers.crf import CRF
#from crf_keras import CRF


#离散
#大纲级别outlinelevel（0、1、2）3
#WordObjectType（null、picture、table、embeddedOLELbject、Omath）5
#IMarkersType（(n),[n],n),L1,L2,L3） 6
#Keyword （keywords abastract、null、figure、table） 5
#italic 斜体 （0\1\mixed） 3
#bold 粗体(0、1、mixed) 3
#IMarkersPos
#连续 数值和0和1
#fontsize、characount、sentencount、linespacing、firstlineindent、spacebefore、spaceafter、wordobjectwidth、wordobjectheight
#Punctuation。

#连续特征共有11个
#离散特征：大纲级别、Word对象类型、标记类型、关键字、粗体、斜体、编号位置。

#全局变量
yaml_file = 'BZ1101'
max_par_num = 1300
num_class = 26 #分类数
#连续特征
continuous_feature_num= 10
#离散特征
#大纲级别
max_outlineLevel = 8
weight_outlineLevel = np.identity(max_outlineLevel)
#Word对象类型
max_wordobjectType = 7
weight_wordobjectType = np.identity(max_wordobjectType)
#标记类型
max_imarkersType = 13
weight_imarkersType = np.identity(max_imarkersType)
#关键字
max_keyWord = 14
weight_keyWord  = np.identity(max_keyWord)
#粗体
max_bold = 4
weight_bold = np.identity(max_bold)
#斜体
max_italic = 5
weight_italic = np.identity(max_italic)

#标记位置
max_imarkerspos = 5
weight_imarkerspos = np.identity(max_imarkerspos)

#颜色
max_fontcolor = 6
weight_fontcolor = np.identity(max_fontcolor)

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
    fontcolor = toArray(fontcolor_value, 'fontcolor')
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
        'fontcolor':fontcolor
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
    emb_outlinelevel = Embedding(max_outlineLevel, max_outlineLevel, mask_zero=True, weights=[weight_outlineLevel])(
        input_outlinelevel)
    #word对象类型
    input_wordobjecttype = Input((max_par_num,), name='wordobjecttype')
    emb_wordobjecttype = Embedding(max_wordobjectType, max_wordobjectType, mask_zero=True, weights=[weight_wordobjectType])(
        input_wordobjecttype)
    #标记类型
    input_imarkerstype = Input((max_par_num,), name='imarkerstype')
    emb_imarkerstype = Embedding(max_imarkersType,max_imarkersType , mask_zero=True, weights=[weight_imarkersType])(input_imarkerstype)
    #关键字
    input_keyword = Input((max_par_num,), name='keyword')
    emb_keyword = Embedding(max_keyWord, max_keyWord, mask_zero=True, weights=[weight_keyWord])(input_keyword)
    #粗体
    input_bold = Input((max_par_num,), name='bold')
    emb_bold = Embedding(max_bold, max_bold, mask_zero=True,weights=[weight_bold])(input_bold)
    #斜体
    input_italic = Input((max_par_num,), name='italic')
    emb_italic = Embedding(max_italic, max_italic, mask_zero=True, weights=[weight_italic])(input_italic)
    #标记位置
    input_imarkerspos = Input((max_par_num,), name='imarkerspos')
    emb_imarkerspos = Embedding(max_imarkerspos, max_imarkerspos, mask_zero=True, weights=[weight_imarkerspos])(input_imarkerspos)

    # 颜色
    input_fontcolor = Input((max_par_num,), name='fontcolor')
    emb_fontcolor = Embedding(max_fontcolor, max_fontcolor, mask_zero=True, weights=[weight_fontcolor])(
        input_fontcolor)

    # print(input_outlineLevel)
    # 最后连接所有这些向量
    input_all = concatenate([input_continuous_feature, emb_outlinelevel, emb_wordobjecttype, emb_imarkerstype, emb_keyword, emb_bold,emb_italic,emb_imarkerspos,emb_fontcolor])

    # print(input_all.shape)#变长序列处理，用Masking层
    input_middle = Masking(mask_value=0)(input_all)

    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2),name='Birectional_gru_1')(input_middle)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.2),name='Birectional_gru_2')(x)
    x = TimeDistributed(Dense(num_class, activation='softmax'))(x)

    # model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
    model = Model(inputs=[input_continuous_feature, input_outlinelevel, input_wordobjecttype, input_imarkerstype, input_keyword,
                          input_bold, input_italic,input_imarkerspos,input_fontcolor], output=x)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    print(model.summary())
    return model


def train():
    #首先调用get_keras_data自定义函数->
    train_data,train_label,_ = get_keras_data('./input/train.csv','train')
    #调自定义的LSTM模型！！！接收双向LSTM网络
    model = GRU_model()
    # 创建一个权重文件保存文件夹logs
    log_dir = "logFiles/"
    # 记录所有训练过程，每隔一定步数记录最大值
    tensorboard = TensorBoard(log_dir=log_dir)
    #Checkpoint是模型的权重
    checkpoint = ModelCheckpoint("./modelFile/{}.hdf5".format(yaml_file),monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #防止过拟合
    early = EarlyStopping(monitor="val_loss",mode="min",patience=20)
    #加了一个tensorboard
    callbacks_list = [tensorboard,checkpoint,early]
    #喂入模型值！
    model.fit(train_data,train_label,epochs=20,validation_split=0.2,batch_size=4,shuffle=False,callbacks=callbacks_list)
    #保存模型
    yaml_string = model.to_yaml()
    with open('./modelFile/{}.yml'.format(yaml_file), 'w') as modelfile:
        modelfile.write(yaml.dump(yaml_string, default_flow_style=True))

if __name__ == '__main__':
    train()