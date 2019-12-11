package com.jyt.error;

/**
 * Created by jyt on 2019/12/5.
 */
public interface CommonError {
    public int getErrCode();
    public String getErrMsg();
    public CommonError setErrMsg(String errMsg);

}
