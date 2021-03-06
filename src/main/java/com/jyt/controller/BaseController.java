package com.jyt.controller;

import com.jyt.error.BusinessException;
import com.jyt.error.EmBusinessError;
import com.jyt.response.CommonReturnType;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.ResponseStatus;

import javax.servlet.http.HttpServletRequest;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by jyt on 2019/12/5.
 */
public class BaseController {

    //定义exceptionhandler解决未被controller层吸收的exception
    //controller异常是业务处理的最后一道关口，若异常被处理掉，返回在前端之前会有一个很好的钩子
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody
    public Object handleException(HttpServletRequest request, Exception ex){
        Map<String,Object> responseData = new HashMap<>();
        if(ex instanceof BusinessException){
            BusinessException businessException =(BusinessException)ex;
            responseData.put("errCode",businessException.getErrCode());
            responseData.put("errMsg",businessException.getErrMsg());

//        CommonReturnType commonReturnType = new CommonReturnType();
//        commonReturnType.setStatus("fail");
//        commonReturnType.setData(responseData);
//        return commonReturnType;
        }else{
            responseData.put("errCode", EmBusinessError.UNKNOWN_ERROR.getErrCode());
            responseData.put("errMsg",EmBusinessError.UNKNOWN_ERROR.getErrMsg());

        }

        return CommonReturnType.create(responseData,"fail");

    }
}
