package com.jyt;

import com.jyt.dao.FileDoMapper;
import com.jyt.dataobject.FileDo;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Hello world!
 *
 */

@SpringBootApplication(scanBasePackages = {"com.jyt"})
@RestController
@MapperScan("com.jyt.dao")
public class App 
{

    @Autowired
    private FileDoMapper fileDoMapper;


    @RequestMapping("/")
    public String home()
    {
        FileDo fileDo= fileDoMapper.selectByPrimaryKey(1);
        if(fileDo == null){
            return "不存在";
        }else{
            return fileDo.getLabel();
        }

    }

    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
        SpringApplication.run(App.class,args);
    }
}
