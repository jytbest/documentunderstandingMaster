package com.jyt.service;

import com.jyt.error.BusinessException;
import com.jyt.service.model.FileModel;

import java.util.List;

/**
 * Created by jyt on 2019/12/5.
 */
public interface FileService {
    //通过id获取file对象的方法
    FileModel getFileById(Integer id);

    //获取全部的File列表数据
    List<FileModel> getAllFile();

    //保存待测文档
    FileModel saveolddocx(FileModel fileModel) throws BusinessException;

    //更新待测文档标签
    FileModel updatelabel(FileModel fileModel)throws BusinessException;

    //更新txt文档标签
    FileModel updatetxt(FileModel fileModel)throws BusinessException;

    //更新newdocx文档
    FileModel updatenewdocx(FileModel fileModel) throws BusinessException;
}
