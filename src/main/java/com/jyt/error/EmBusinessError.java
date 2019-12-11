package com.jyt.error;

/**
 * Created by jyt on 2019/12/5.
 */
public enum EmBusinessError implements CommonError {
    //通用错误码
    PARAMETER_VALIDATION_ERROR(10001,"参数不合法"),
    UNKNOWN_ERROR(10002,"未知错误"),

    //1000开头为文件信息相关错误定义
    FILE_NOT_EXIST(20001,"文件不存在"),
    //1000开头为文件信息相关错误定义
    TIME_NOT_EXIST(20003,"时间不存在"),

    //1000开头为文件信息相关错误定义
    DATA_NOT_IN_MYSQL(20002,"数据库暂无簇模型数据"),


    OLDDOCXFILE_FAIL_SAVE(30001,"待测文件保存失败！"),
    OLDDOCXLABEL_FAIL_SAVE(30002,"待测文件所在簇标签保存失败！"),
    TXT_FAIL_UPDATE(30003,"特征文件保存失败！"),
    NEWDOCX_FAIL_UPDATE(30004,"标注文件保存失败！")
    ;

    private EmBusinessError(int errCode,String errMsg){
        this.errCode=errCode;
        this.errMsg=errMsg;

    }


    private int errCode;
    private String errMsg;



    @Override
    public int getErrCode() {
        return this.errCode;
    }

    @Override
    public String getErrMsg() {
        return this.errMsg;
    }

    @Override
    public CommonError setErrMsg(String errMsg) {
        this.errMsg = errMsg;

        return this;
    }
}
