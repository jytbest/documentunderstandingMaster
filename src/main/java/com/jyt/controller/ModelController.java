package com.jyt.controller;

import com.jyt.controller.viewobject.ModelVO;
import com.jyt.error.BusinessException;
import com.jyt.error.EmBusinessError;
import com.jyt.response.CommonReturnType;
import com.jyt.service.ModelService;
import com.jyt.service.model.ModelModel;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jyt on 2019/12/6.
 */

@Controller("model")
@RequestMapping("/model")
@CrossOrigin
public class ModelController extends BaseController {

    @Autowired
    private ModelService modelService;



    //查询所有的模型
    @RequestMapping("/list")
    @ResponseBody
    public CommonReturnType modellist() throws BusinessException {
        List<ModelModel> modelModelList = modelService.modellist();
        List<ModelVO> modelVOList = modelModelList.stream().map( modelModel -> {
            ModelVO modelVO = this.convertfromModel(modelModel);
            return modelVO;
        }).collect(Collectors.toList());
       if(modelVOList == null){
            throw new BusinessException(EmBusinessError.DATA_NOT_IN_MYSQL);
        }
        return CommonReturnType.create(modelVOList);
    }

    //转类型
    private ModelVO convertfromModel(ModelModel modelModel){
        if(modelModel == null){
            return null;
        }
        ModelVO modelVO = new ModelVO();
        BeanUtils.copyProperties(modelModel,modelVO);
        return modelVO;
    }
}
