package com.jyt.error;

/**
 * Created by jyt on 2019/12/5.
 */


//设计模式，包装器业务异常类实现！！！两个类共同继承CommonError对应的方法，
// 无论是new BusinessException 还是new EmBusinessError,都可以有errorCode和errorMsg的组装定义
// 并且需要共同实现setErrmsg方法
public class BusinessException extends Exception implements CommonError{

    private CommonError commonError;


    //直接接收EmBusinessError的传参用于构造业务异常
    public BusinessException(CommonError commonError){
        super();
        this.commonError = commonError;
    }

    //接收自定义errMsg的方式构造业务异常
    public BusinessException(CommonError commonError,String errMsg){
        super();
        this.commonError = commonError;
        this.commonError.setErrMsg(errMsg);
    }


    @Override
    public int getErrCode() {
        return this.commonError.getErrCode();
    }

    @Override
    public String getErrMsg() {
        return this.commonError.getErrMsg();
    }

    @Override
    public CommonError setErrMsg(String errMsg) {
        this.commonError.setErrMsg(errMsg);

        return this;
    }
}
