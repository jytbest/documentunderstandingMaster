package com.jyt.service.impl;

import com.alibaba.druid.util.StringUtils;
import com.jyt.dao.FileDoMapper;
import com.jyt.dataobject.FileDo;
import com.jyt.error.BusinessException;
import com.jyt.error.EmBusinessError;
import com.jyt.service.FileService;
import com.jyt.service.model.FileModel;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by jyt on 2019/12/5.
 */
@Service
public class FileServiceImpl implements FileService {

    @Autowired
    private FileDoMapper fileDoMapper;




    //获取所有的filelist
    @Override
    public List<FileModel> getAllFile() {
        List<FileDo> fileDoList = fileDoMapper.findAll();
        List<FileModel> fileModelList = fileDoList.stream().map(fileDo -> {
            FileModel fileModel = this.convertFromDataObject(fileDo);
            return fileModel;
        }).collect(Collectors.toList());
        return fileModelList;
    }


    //保存待测文档(当有两个insert操作，需要开启事务注解)
    @Override
    @Transactional
    public FileModel saveolddocx(FileModel fileModel) throws BusinessException {

        if(fileModel.getOlddocx() == null){
            throw new BusinessException(EmBusinessError.FILE_NOT_EXIST);
        }else if(fileModel.getTime() == null){
            throw new BusinessException(EmBusinessError.TIME_NOT_EXIST);
        }

        FileDo fileDo = convertFromModel(fileModel);

        fileDoMapper.insertSelective(fileDo);

        fileModel.setFileid(fileDo.getFileid());

        return this.getFileById(fileModel.getFileid());

    }

    @Override
    public FileModel getFileById(Integer fileid) {
        FileDo fileDo = fileDoMapper.selectByPrimaryKey(fileid);
        if(fileDo == null){
            return null;
        }
        return convertFromDataObject(fileDo);

    }




    //更新待测文档标签
    @Override
    @Transactional
    public FileModel updatelabel(FileModel fileModel) throws BusinessException {
        if(StringUtils.isEmpty(fileModel.getLabel())){
            throw new BusinessException(EmBusinessError.OLDDOCXLABEL_FAIL_SAVE);
        }
        FileDo fileDo = convertFromModel(fileModel);
        //实现model->转成dataobject方法
        fileDoMapper.updateByPrimaryKeySelective(fileDo);

        return this.getFileById(fileModel.getFileid());

    }

    //更新txt文档标签
    @Override
    public FileModel updatetxt(FileModel fileModel) throws BusinessException {
        if(StringUtils.isEmpty(fileModel.getFeaturetxt())){
            throw new BusinessException(EmBusinessError.TXT_FAIL_UPDATE);
        }
        FileDo fileDo = convertFromModel(fileModel);
        //实现model->转成dataobject方法
        fileDoMapper.updateByPrimaryKeySelective(fileDo);

        return this.getFileById(fileModel.getFileid());
    }


    //更新newdocx文档
    @Override
    public FileModel updatenewdocx(FileModel fileModel) throws BusinessException {
        if(StringUtils.isEmpty(fileModel.getNewdocx())){
            throw new BusinessException(EmBusinessError.NEWDOCX_FAIL_UPDATE);
        }

        FileDo fileDo = convertFromModel(fileModel);
        //实现model->转成dataobject方法
        fileDoMapper.updateByPrimaryKeySelective(fileDo);

        return this.getFileById(fileModel.getFileid());
    }


    private FileDo convertFromModel(FileModel fileModel){
        if(fileModel ==null){
            return null;
        }
        FileDo fileDo = new FileDo();
        BeanUtils.copyProperties(fileModel,fileDo);
        return fileDo;
    }

    private FileModel convertFromDataObject(FileDo fileDo){
        if(fileDo == null){
            return null;
        }
        FileModel fileModel = new FileModel();
        BeanUtils.copyProperties(fileDo,fileModel);
        return fileModel;
    }
}
