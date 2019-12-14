package com.jyt.controller;

import com.jyt.App;
import com.spire.doc.Document;
import com.spire.doc.FileFormat;
import com.spire.doc.Section;
import com.spire.doc.collections.CommentsCollection;
import com.spire.doc.documents.Paragraph;
import com.spire.doc.fields.Comment;

import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import jdk.nashorn.internal.runtime.regexp.joni.Regex;
import org.springframework.boot.SpringApplication;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by jyt on 2019/12/13.
 */
@Controller("Extract")
@RequestMapping("/tools")
@CrossOrigin
public class ExtractController {
//    @RequestMapping("/extract")
//    @ResponseBody
    public String extractfeature() {
        //Document document = new Document("/Users/jyt/Project/documentunderstanding/file/olddocx/A20170922090728222.docx");
        Document document = new Document("/Users/jyt/Project/documentunderstanding/file/olddocx/test.docx");

        int j = 0;
        int commentIndex = 1;
        for (int i = 0; i < document.getSections().getCount(); i++) {
            Section section = document.getSections().get(i);
            while (j < section.getParagraphs().getCount()) {
                Paragraph paragraph = section.getParagraphs().get(j);
                Pattern p = Pattern.compile("\\s{" + paragraph.getText().length() + ",}");
                Matcher m = p.matcher(paragraph.getText());
                m.replaceAll("");
                //添加批注
                if (!(paragraph.getWordCount() == 0)) {
                    //System.out.println(paragraph.getText());
                    Comment comment = paragraph.appendComment("NO."+commentIndex);
                    comment.getFormat().setAuthor("JYTCrystal");
                    comment.getFormat().setInitial("CM");
                    commentIndex++;
                }
                j++;
            }
        }
        String root = System.getProperty("user.home")+ File.separator+"Project";
        String rootpath = root+"/documentunderstanding/file/feature/comment"+new Date()+".docx";
        //保存文档
        document.saveToFile(rootpath, FileFormat.Docx);
        return rootpath;

    }

    public Map<String, String> getFeatureSet(Document doc, int commentIndex){
        Map<String,String> dic = new HashMap<String,String>();
        Comment comm = doc.getComments().get(commentIndex);
        String text0 = "";
        //dic.put("TextContent", getTextContent(comm));
        dic.put("FontName", getFontName(comm));
        dic.put("FontSize", getFontSize(comm));
        dic.put("KeyWord",getKeyWord(comm));
        //dic.put("Bold")

        System.out.println(dic);
        return dic;

    }

    //提字号特征
    private String getFontSize(Comment comm){
        float fontSize = comm.getOwnerParagraph().getBreakCharacterFormat().getFontSize();
        return Float.toString(fontSize);
    }
    //提关键字
    private String getKeyWord(Comment comm){
        String txt = comm.getOwnerParagraph().getText().trim();
        String keyword = "null";
        ArrayList keywordsList = new ArrayList();
        keywordsList.add("(^([\\s]*)摘([\\s]*)要([\\s]*))");//正则表达式——摘要
        keywordsList.add("(^([\\s]*)abstract([\\s]*))");//正则表达式——摘要
        keywordsList.add("(^([\\s]*)关([\\s]*)键([\\s]*)字([\\s]*))");//正则表达式——关键字
        keywordsList.add("(^([\\s]*)关([\\s]*)键([\\s]*)词([\\s]*))");//正则表达式——关键词
        keywordsList.add("(^([\\s]*)key([\\s]*)words([\\s]*))");//正则表达式——关键词
        keywordsList.add("^([\\s]*)图([\\s]*)([0-9]*)");//正则表达式——图题
        keywordsList.add("^([\\s]*)表([\\s]*)([0-9]*)");//正则表达式——表题
        keywordsList.add("^([\\s]*)figure([\\s]*)([0-9]*)");//正则表达式——
        keywordsList.add("^([\\s]*)table([\\s]*)([0-9]*)");//正则表达式——
        keywordsList.add("(^([\\s]*)参([\\s]*)考([\\s]*)文([\\s]*)献)");//正则表达式——
        //keywordsList.Add(new Regex("(^([\\s]*)引([\\s]*)言)"));//正则表达式——关键词
        //keywordsList.Add(new Regex("(^([\\s]*)结([\\s]*)束([\\s]*)语)"));//正则表达式——关键词
        for (int keywordsList_i = 0; keywordsList_i < keywordsList.size(); keywordsList_i++) {
            String regex = (String) keywordsList.get(keywordsList_i);

            Pattern p = Pattern.compile(regex);
            Matcher m = p.matcher(txt.toLowerCase());
            String str="";
            Boolean is = false;
            while(m.find()) {
                str = m.group(0).replace(" ","").replace("  ","").replace("\t","");
                is = true;
            }
            if(is){
                keyword = str.replaceAll("\\d", "");
                System.out.println(keyword);
                break;
            }
        }
        System.out.println(txt);
        String regexemail = "^[A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\\.[a-zA-Z0-9_-]+)+$";
        if(txt.matches(regexemail)) keyword = "邮箱";
        return keyword.replace(",","");
    }

    //提字体特征
    private String getFontName(Comment comm){
        String  fontName = comm.getOwnerParagraph().getBreakCharacterFormat().getFontName();
        if(fontName.equals("")){
            fontName = comm.getOwnerParagraph().get(1).getCharacterFormat().getFontName();
        }
        return  fontName.replace(",","");
    }

    //提颜色特征



    public void Featurestotxt(Document doc,String txtpath){
        CommentsCollection commons = doc.getComments();
        String[] FontSize = new String[500];
        String[] KeyWord = new String[500];
        String[] FontName = new String[500];
        for(int i=0 ;i < commons.counts();i++){
            Map<String, String> features = getFeatureSet(doc, i);
            FontSize[i] = features.get("FontSize");
            KeyWord[i] = features.get("KeyWord");
            FontName[i] =features.get("FontName");
            System.out.println(i+":"+FontSize[i]);
            System.out.println(KeyWord[i]);
            FontName[i] =features.get("FontName");

        }
    }

    @RequestMapping("/begin")
    @ResponseBody
    public void begin(){
        String rootpath = extractfeature();
        Document doc = new Document(rootpath);
        String root = System.getProperty("user.home")+ File.separator+"Project";
        String txtpath = root+"/documentunderstanding/file/feature/comment.txt";
        Featurestotxt(doc,txtpath);
        return ;
    }


    @RequestMapping("/todo")
    @ResponseBody
    public void test() {
        String txt = "关键词:茶馆；node.js；HTML；javascript；mongodb；预定；查询 ；";
        String regex= "(^([\\s]*)关([\\s]*)键([\\s]*)词([\\s]*[(\\:)|(：)])([\\s\\S]*))";
        boolean b = txt.matches(regex);
        System.out.println(b);
        if (txt.matches(regex))//正则表达式匹配
        {
            Pattern p = Pattern.compile(regex);
            Matcher m = p.matcher(txt.toLowerCase());
            String str = "";
            while (m.find()) {
                str = m.group(0).replace(" ", "").replace("  ", "").replace("\t", "");
            }
            System.out.println(str);
        }
    }

}
