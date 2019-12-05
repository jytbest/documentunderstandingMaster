package com.jyt.service;

import com.jyt.service.model.FileModel;

/**
 * Created by jyt on 2019/12/5.
 */
public interface FileService {
    //通过id获取file对象的方法
    FileModel getFileById(Integer id);
}
