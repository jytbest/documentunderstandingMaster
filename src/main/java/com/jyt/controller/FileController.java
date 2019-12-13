package com.jyt.controller;

import com.jyt.controller.viewobject.FileVO;
import com.jyt.error.BusinessException;
import com.jyt.error.EmBusinessError;
import com.jyt.response.CommonReturnType;
import com.jyt.service.FileService;
import com.jyt.service.model.FileModel;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.awt.*;
import java.io.*;
import java.sql.Timestamp;
import java.util.Date;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.spire.doc.*;

import com.spire.doc.documents.Paragraph;

import com.spire.doc.fields.Comment;


/**
 * Created by jyt on 2019/12/5.
 */

@Controller("file")
@RequestMapping("/report")
@CrossOrigin
public class FileController extends BaseController {

    @Autowired
    private FileService fileService;



    @RequestMapping("/list")
    @ResponseBody
    public CommonReturnType getAllFill() throws BusinessException {
       List<FileModel> allfileModel = fileService.getAllFile();
       List<FileVO> allfileVO = allfileModel.stream().map(fileModel->{
           FileVO fileVO =convertFromModel(fileModel);
           return fileVO;
       }).collect(Collectors.toList());

       if(allfileModel == null){
            //allfileModel.setLabel("111");
            throw new BusinessException(EmBusinessError.DATA_NOT_IN_MYSQL);
        }

        return CommonReturnType.create(allfileVO);
    }


    //后端接收文件保存到文件夹并将路径保存到数据库中。docx
    @RequestMapping("/saveolddocx")
    @ResponseBody
    public CommonReturnType saveOlddocx(@RequestParam(name= "file") MultipartFile olddocx) throws BusinessException, IOException {

        //获取到待测的docx文件，然后保存到指定路径
        //获取文件名（包括后缀）
        String root = System.getProperty("user.home")+File.separator+"Project";
        System.out.println(root);
        String rootpath = root+"/documentunderstanding/file/olddocx";
        File file = new File(rootpath);
        if (!file.exists()) { System.out.println("错误！");}

        String filename = olddocx.getOriginalFilename();
        String path = rootpath+"/"+filename;

       //调Python脚本需要的路径
        String ppath = root+"/documentunderstanding/cluster1/testfile/test1210.docx";

        FileOutputStream fos = null;
        FileOutputStream pfos = null;
        try{
            fos = new FileOutputStream(path);
            fos.write(olddocx.getBytes());
            pfos = new FileOutputStream(ppath,false);
            pfos.write(olddocx.getBytes());
            System.out.println("保存成功！");
        }catch (Exception e){
            throw new BusinessException(EmBusinessError.OLDDOCXFILE_FAIL_SAVE);
        }finally {
                fos.close();
                pfos.close();

                //保存待测文档的流程

                Date date = new Date();
                Timestamp t = new Timestamp(date.getTime());
                System.out.println(t);
                FileModel fileModel = new FileModel();
                //保存的路径是不会被覆盖的！！！
                fileModel.setOlddocx(path); //注意数据库中字段的规格，设置最长的字符数
                fileModel.setTime(t);
                FileModel fileModelForReturn =fileService.saveolddocx(fileModel);
                FileVO fileVO = convertFromModel(fileModelForReturn);
                return CommonReturnType.create(fileVO);
        }
        //return CommonReturnType.create(fileVO);
    }



    //调用聚类模型，得到当前文档的类型结果，然后更新到数据库
    @RequestMapping("/updatelabel")
    @ResponseBody
    public CommonReturnType updatelabel(@RequestParam(name="fileid") Integer id) throws BusinessException {
        /*
        调用聚类模型，执行cmd命令，最后得到待测文档所在的簇
        */
        String label = execmdcluster();
        System.out.println(label);
        //测试完成，无bug!!!
        //更新到数据库
        FileModel fileModel = new FileModel();
        fileModel.setLabel(label);
        fileModel.setFileid(id);
        FileModel fileModelForReturn = fileService.updatelabel(fileModel);

        FileVO fileVO = convertFromModel(fileModelForReturn);

        return CommonReturnType.create(fileVO);
    }


    //后端接收文件保存到文件夹并将路径更新到数据库中。txt
    @RequestMapping("/updatetxt")
    @ResponseBody
    public CommonReturnType updatetxt(@RequestParam(name="txtfile") MultipartFile txt,
                                      @RequestParam(name="label") String label,
                                      @RequestParam(name="fileid") Integer id) throws BusinessException {


        String root = System.getProperty("user.home")+File.separator+"Project";
        System.out.println(root);
        //获取到txt文件，然后保存到指定路径
        String rootpath = root+"/documentunderstanding/file/csv";
        File file = new File(rootpath);
        if (!file.exists()) { System.out.println("错误！");}
        //获取文件名（包括后缀）
        String filename = txt.getOriginalFilename();
        //文件目录
        String path = rootpath+"/"+filename;
        //调簇分类器需要的路径
        String grupath = root+"/documentunderstanding/gru/"+label+"/input/prediction.csv";
        FileOutputStream fos = null;
        FileOutputStream pfos = null;
        try{
            fos = new FileOutputStream(path);
            fos.write(txt.getBytes());
            //覆盖原有内容
            pfos = new FileOutputStream(grupath,false);
            pfos.write(txt.getBytes());
            System.out.println("保存成功！");
        }catch (Exception e){
            throw new BusinessException(EmBusinessError.TXT_FAIL_UPDATE);
        }finally {
            try {
                fos.close();
                pfos.close();
                //保存txt文档的流程
                FileModel fileModel = new FileModel();
                //这个是不会覆盖的TXT路径
                fileModel.setFeaturetxt(path);
                fileModel.setFileid(id);
                FileModel fileModelForReturn = fileService.updatetxt(fileModel);
                FileVO fileVO = convertFromModel(fileModelForReturn);
                return CommonReturnType.create(fileVO);
            } catch (IOException e) {
                throw new BusinessException(EmBusinessError.TXT_FAIL_UPDATE);
            }
        }
    }


    //调用对应文件夹下的gru模型，并得到result.txt结果，后端接收文件保存到文件夹并将路径更新到数据库中。newdocx
    @RequestMapping("/updatereport")
    @ResponseBody
    public CommonReturnType updatereport(@RequestParam(name="fileid") Integer id,
                                         @RequestParam(name="label") String label,
                                         @RequestParam(name="olddocx") String olddocxpath) throws BusinessException {
        //调用gru预测模型，最后得到待测文档的result文件
        String root = System.getProperty("user.home")+File.separator+"Project";
        System.out.println(root);

        execmdBigru(label);
        String gruoutput= root+"/documentunderstanding/gru/"+label+"/output/predict.csv";

        String reportpath = markup(olddocxpath,gruoutput);

        //测试完成！无bug!
        //String path = "/啦啦啦啦啦";
        //后端接收文件，保存到文件夹，并将路径更新到数据库中newdocx
        FileModel fileModel = new FileModel();
        fileModel.setNewdocx(reportpath);
        fileModel.setFileid(id);
        FileModel fileModelForReturn = fileService.updatenewdocx(fileModel);
        FileVO fileVO = convertFromModel(fileModelForReturn);

        return CommonReturnType.create(fileVO);
    }



    @RequestMapping("/get")
    @ResponseBody
    public CommonReturnType getFile(@RequestParam(name="id") Integer id) throws BusinessException {
        //调用service服务获取对应id的file对象并返回给前端
        FileModel fileModel = fileService.getFileById(id);

        //若获取的信息不存在
        if(fileModel == null){
            //fileModel.setLabel("111");
            throw new BusinessException(EmBusinessError.FILE_NOT_EXIST);
        }
        //拦截掉Tomcat异常500的处理的方式，去解决问题

        //将核心领域模型对象转化为可供UI使用的viewobject
        FileVO fileVO = convertFromModel(fileModel);
        //CommonReturnType静态的不用new
        //返回通用对象
        return CommonReturnType.create(fileVO);
    }


    private String execmdcluster(){
        //String cmd1= "cd /Users/jyt/Downloads";
        String cmd = "python3 run.py";
        String label = null;
        try {
            //Runtime.getRuntime().exec(cmd1);
            Process process = Runtime.getRuntime().exec(cmd,null,new File("/Users/jyt/Project/documentunderstanding/cluster1/kmeans"));
            InputStream is = process.getInputStream();
            InputStreamReader isr = new InputStreamReader(is);
            BufferedReader br = new BufferedReader(isr);
            String[] array= new String[8];
            String content;
            int i =0;
            while ((content=br.readLine())!= null) {
                array[i]=content;
                i++;
            }
            System.out.println(array[2]);
            String rgex ="#(.*)";
            Pattern pattern = Pattern.compile(rgex);
            Matcher matcher = pattern.matcher(array[2]);
            while (matcher.find()) {
                //System.out.println(matcher.group(1));
                label = matcher.group(1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println(label);
        return label;
    }

    @RequestMapping("/delete")
    @ResponseBody
    public CommonReturnType delete(@RequestParam(name="fileid") Integer fileid) throws BusinessException {
        System.out.println(fileid);
        Integer flag = fileService.deletebyfileid(fileid);

        return CommonReturnType.create(flag);
    }

    private void execmdBigru(String label){
        //String cmd1= "cd /Users/jyt/Downloads";
        String cmd = "python3 predict.py";
        try {
            //Runtime.getRuntime().exec(cmd1);
            Runtime.getRuntime().exec(cmd,null,new File("/Users/jyt/Project/documentunderstanding/gru/"+label));

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    @RequestMapping("opendocx")
    @ResponseBody
    public String opendocx(@RequestParam(name = "path") String path) throws IOException {
        String[] cmd=new String[]{"open","-a","/Applications/Microsoft Word.app",path};
        //String cmd = "open -a /Applications/MicrosoftWor.app"+" "+path;
        try {
            //Runtime.getRuntime().exec(cmd1);
            Runtime.getRuntime().exec(cmd);
        } catch (IOException e) {
            e.printStackTrace();
        }
     return "成功";
    }

    @RequestMapping("openother")
    @ResponseBody
    public String openother(@RequestParam(name = "path") String path) throws IOException {
        String[] cmd=new String[]{"open","-a","/Applications/Sublime Text.app",path};
        try {

            Runtime.getRuntime().exec(cmd);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "成功";
    }

//    @RequestMapping("markup")
//    @ResponseBody
    private String markup(String olddocxpath,String csvpath){
        //加载需要添加批注的word文档
        Document document = new Document(olddocxpath);


        String[] csvstr = csvtostr(csvpath);

        int j =0;
        int commentIndex = 1;
        for(int i = 0; i < document.getSections().getCount(); i++) {
            Section section = document.getSections().get(i);
            while(j < section.getParagraphs().getCount()){
                Paragraph paragraph = section.getParagraphs().get(j);
                Pattern p = Pattern.compile("\\s{"+paragraph.getText().length()+",}");
                Matcher m = p.matcher(paragraph.getText());
                m.replaceAll("");
                //添加批注

                if(!(paragraph.getWordCount()==0)){
                    //System.out.println(paragraph.getText());
                    Comment comment = paragraph.appendComment(csvstr[commentIndex]);
                    comment.getFormat().setAuthor("JYTCrystal");
                    comment.getFormat().setInitial("CM");
                    commentIndex++;
                }
                j++;
            }
        }


        String root = System.getProperty("user.home")+File.separator+"Project";
        System.out.println(root);
        //获取到txt文件，然后保存到指定路径
        String rootpath = root+"/documentunderstanding/file/report/markup"+new Date()+".docx";

        //保存文档
        document.saveToFile(rootpath, FileFormat.Docx);

        return rootpath;
    }
//    @RequestMapping("csvstr")
//    @ResponseBody
    private String[] csvtostr(String path){
        String[] array= new String[500];
        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            String strLine = null;
            int lineNumber = 0;
            String context;
            while ((context = br.readLine()) != null) {
                array[lineNumber]= context;
                //System.out.println(array[lineNumber]);
                lineNumber++;
            }
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        //System.out.println(array);
        return array;
    }


    private FileVO convertFromModel(FileModel fileModel){
        if(fileModel==null){
            return null;
        }
        FileVO fileVO = new FileVO();
        BeanUtils.copyProperties(fileModel,fileVO);
        return fileVO;
    }


}
