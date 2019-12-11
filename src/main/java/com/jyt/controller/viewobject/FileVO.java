package com.jyt.controller.viewobject;

import java.util.Date;

/**
 * Created by jyt on 2019/12/5.
 */
public class FileVO {
    private Integer fileid;

    private Date time;

    private String olddocx;

    private String label;

    private String featuretxt;

    private String newdocx;

    public String getNewdocx() {
        return newdocx;
    }

    public void setNewdocx(String newdocx) {
        this.newdocx = newdocx;
    }


    public Integer getFileid() {
        return fileid;
    }

    public void setFileid(Integer fileid) {
        this.fileid = fileid;
    }

    public Date getTime() {
        return time;
    }

    public void setTime(Date time) {
        this.time = time;
    }

    public String getOlddocx() {
        return olddocx;
    }

    public void setOlddocx(String olddocx) {
        this.olddocx = olddocx;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getFeaturetxt() {
        return featuretxt;
    }

    public void setFeaturetxt(String featuretxt) {
        this.featuretxt = featuretxt;
    }

}
