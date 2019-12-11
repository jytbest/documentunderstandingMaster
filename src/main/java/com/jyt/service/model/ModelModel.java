package com.jyt.service.model;

/**
 * Created by jyt on 2019/12/6.
 */
public class ModelModel {
    private Integer modelid;

    private String label;

    private String description;

    private String modelcol;

    private String rolecollection;

    private String traindata;

    private String execode;

    private String hdf5;

    private String yml;

    public Integer getModelid() {
        return modelid;
    }

    public void setModelid(Integer modelid) {
        this.modelid = modelid;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public String getModelcol() {
        return modelcol;
    }

    public void setModelcol(String modelcol) {
        this.modelcol = modelcol;
    }

    public String getRolecollection() {
        return rolecollection;
    }

    public void setRolecollection(String rolecollection) {
        this.rolecollection = rolecollection;
    }

    public String getTraindata() {
        return traindata;
    }

    public void setTraindata(String traindata) {
        this.traindata = traindata;
    }

    public String getExecode() {
        return execode;
    }

    public void setExecode(String execode) {
        this.execode = execode;
    }

    public String getHdf5() {
        return hdf5;
    }

    public void setHdf5(String hdf5) {
        this.hdf5 = hdf5;
    }

    public String getYml() {
        return yml;
    }

    public void setYml(String yml) {
        this.yml = yml;
    }
}
