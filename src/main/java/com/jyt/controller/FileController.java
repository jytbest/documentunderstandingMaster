package com.jyt.controller;

import com.jyt.service.FileService;
import com.jyt.service.model.FileModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * Created by jyt on 2019/12/5.
 */

@Controller("file")
@RequestMapping("/file")
public class FileController {
    @Autowired
    private FileService fileService;



    @RequestMapping("/get")
    @ResponseBody
    public FileModel getFile(@RequestParam(name="id") Integer id){
        //调用service服务获取对应id的file对象并返回给前端
        FileModel fileModel = fileService.getFileById(id);

        return fileModel;
    }
}
