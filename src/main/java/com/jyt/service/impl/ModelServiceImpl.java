package com.jyt.service.impl;

import com.jyt.dao.ModelDoMapper;
import com.jyt.dataobject.ModelDo;
import com.jyt.service.ModelService;
import com.jyt.service.model.FileModel;
import com.jyt.service.model.ModelModel;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jyt on 2019/12/6.
 */

@Service
public class ModelServiceImpl implements ModelService {

    @Autowired
    private ModelDoMapper modelDoMapper;

    @Override
    public List<ModelModel> modellist() {
        List<ModelDo> modelDoList = modelDoMapper.findAll();
        List<ModelModel> modelModelList =modelDoList.stream().map(modelDo -> {
            ModelModel modelModel = this.convertFromDataObject(modelDo);
            return modelModel;
        }).collect(Collectors.toList());

        return modelModelList;
    }

    private ModelModel convertFromDataObject(ModelDo modelDo){
        if(modelDo == null){
            return null;
        }
        ModelModel modelModel = new ModelModel();
        BeanUtils.copyProperties(modelDo,modelModel);

        return modelModel;
    }
}
