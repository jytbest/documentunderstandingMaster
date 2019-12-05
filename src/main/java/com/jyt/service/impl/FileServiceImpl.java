package com.jyt.service.impl;

import com.jyt.dao.FileDoMapper;
import com.jyt.dataobject.FileDo;
import com.jyt.service.FileService;
import com.jyt.service.model.FileModel;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Created by jyt on 2019/12/5.
 */
@Service
public class FileServiceImpl implements FileService {

    @Autowired
    private FileDoMapper fileDoMapper;



    @Override
    public FileModel getFileById(Integer id) {
        FileDo fileDo = fileDoMapper.selectByPrimaryKey(id);
        if(fileDo == null){
            return null;
        }
        return convertFromDataObject(fileDo);

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
