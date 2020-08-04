
package hopes.cic.xml;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for TaskType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="TaskType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="dataParallel" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskDataParallelType" minOccurs="0"/>
 *         &lt;element name="port" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskPortType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="multicastPort" type="{http://peace.snu.ac.kr/CICXMLSchema}MulticastPortType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="mode" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskModeType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="libraryMasterPort" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryMasterPortType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="parameter" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskParameterType" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="extraHeader" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="extraSource" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="extraCIC" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="extraFile" type="{http://www.w3.org/2001/XMLSchema}string" maxOccurs="unbounded" minOccurs="0"/>
 *         &lt;element name="mtm" type="{http://peace.snu.ac.kr/CICXMLSchema}MTMType" minOccurs="0"/>
 *         &lt;element name="loopStructure" type="{http://peace.snu.ac.kr/CICXMLSchema}LoopStructureType" minOccurs="0"/>
 *         &lt;element name="hardwareDependency" type="{http://peace.snu.ac.kr/CICXMLSchema}HardwareDependencyType" minOccurs="0"/>
 *         &lt;element name="faultTolerance" type="{http://peace.snu.ac.kr/CICXMLSchema}FaultToleranceType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="ParentTask" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="cflags" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="description" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="file" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="hasInternalStates" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" />
 *       &lt;attribute name="hasMTM" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="hasSubGraph" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="id" use="required" type="{http://www.w3.org/2001/XMLSchema}nonNegativeInteger" />
 *       &lt;attribute name="isHardwareDependent" type="{http://peace.snu.ac.kr/CICXMLSchema}YesNoType" />
 *       &lt;attribute name="language" type="{http://peace.snu.ac.kr/CICXMLSchema}LanguageType" />
 *       &lt;attribute name="ldflags" type="{http://www.w3.org/2001/XMLSchema}string" />
 *       &lt;attribute name="name" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="runCondition" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}RunConditionType" />
 *       &lt;attribute name="subGraphProperty" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *       &lt;attribute name="taskType" use="required" type="{http://www.w3.org/2001/XMLSchema}string" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "TaskType", propOrder = {
    "dataParallel",
    "port",
    "multicastPort",
    "mode",
    "libraryMasterPort",
    "parameter",
    "extraHeader",
    "extraSource",
    "extraCIC",
    "extraFile",
    "mtm",
    "loopStructure",
    "hardwareDependency",
    "faultTolerance"
})
public class TaskType {

    protected TaskDataParallelType dataParallel;
    protected List<TaskPortType> port;
    protected List<MulticastPortType> multicastPort;
    protected List<TaskModeType> mode;
    protected List<LibraryMasterPortType> libraryMasterPort;
    protected List<TaskParameterType> parameter;
    protected List<String> extraHeader;
    protected List<String> extraSource;
    protected List<String> extraCIC;
    protected List<String> extraFile;
    protected MTMType mtm;
    protected LoopStructureType loopStructure;
    protected HardwareDependencyType hardwareDependency;
    protected FaultToleranceType faultTolerance;
    @XmlAttribute(name = "ParentTask", required = true)
    protected String parentTask;
    @XmlAttribute
    protected String cflags;
    @XmlAttribute
    protected String description;
    @XmlAttribute(required = true)
    protected String file;
    @XmlAttribute
    protected YesNoType hasInternalStates;
    @XmlAttribute
    protected String hasMTM;
    @XmlAttribute
    protected String hasSubGraph;
    @XmlAttribute(required = true)
    protected BigInteger id;
    @XmlAttribute
    protected YesNoType isHardwareDependent;
    @XmlAttribute
    protected String language;
    @XmlAttribute
    protected String ldflags;
    @XmlAttribute(required = true)
    protected String name;
    @XmlAttribute(required = true)
    protected RunConditionType runCondition;
    @XmlAttribute
    protected String subGraphProperty;
    @XmlAttribute(required = true)
    protected String taskType;

    /**
     * Gets the value of the dataParallel property.
     * 
     * @return
     *     possible object is
     *     {@link TaskDataParallelType }
     *     
     */
    public TaskDataParallelType getDataParallel() {
        return dataParallel;
    }

    /**
     * Sets the value of the dataParallel property.
     * 
     * @param value
     *     allowed object is
     *     {@link TaskDataParallelType }
     *     
     */
    public void setDataParallel(TaskDataParallelType value) {
        this.dataParallel = value;
    }

    /**
     * Gets the value of the port property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the port property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getPort().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TaskPortType }
     * 
     * 
     */
    public List<TaskPortType> getPort() {
        if (port == null) {
            port = new ArrayList<TaskPortType>();
        }
        return this.port;
    }

    /**
     * Gets the value of the multicastPort property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the multicastPort property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMulticastPort().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link MulticastPortType }
     * 
     * 
     */
    public List<MulticastPortType> getMulticastPort() {
        if (multicastPort == null) {
            multicastPort = new ArrayList<MulticastPortType>();
        }
        return this.multicastPort;
    }

    /**
     * Gets the value of the mode property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the mode property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getMode().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TaskModeType }
     * 
     * 
     */
    public List<TaskModeType> getMode() {
        if (mode == null) {
            mode = new ArrayList<TaskModeType>();
        }
        return this.mode;
    }

    /**
     * Gets the value of the libraryMasterPort property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the libraryMasterPort property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getLibraryMasterPort().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link LibraryMasterPortType }
     * 
     * 
     */
    public List<LibraryMasterPortType> getLibraryMasterPort() {
        if (libraryMasterPort == null) {
            libraryMasterPort = new ArrayList<LibraryMasterPortType>();
        }
        return this.libraryMasterPort;
    }

    /**
     * Gets the value of the parameter property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the parameter property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getParameter().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link TaskParameterType }
     * 
     * 
     */
    public List<TaskParameterType> getParameter() {
        if (parameter == null) {
            parameter = new ArrayList<TaskParameterType>();
        }
        return this.parameter;
    }

    /**
     * Gets the value of the extraHeader property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraHeader property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraHeader().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraHeader() {
        if (extraHeader == null) {
            extraHeader = new ArrayList<String>();
        }
        return this.extraHeader;
    }

    /**
     * Gets the value of the extraSource property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraSource property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraSource().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraSource() {
        if (extraSource == null) {
            extraSource = new ArrayList<String>();
        }
        return this.extraSource;
    }

    /**
     * Gets the value of the extraCIC property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraCIC property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraCIC().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraCIC() {
        if (extraCIC == null) {
            extraCIC = new ArrayList<String>();
        }
        return this.extraCIC;
    }

    /**
     * Gets the value of the extraFile property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extraFile property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtraFile().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link String }
     * 
     * 
     */
    public List<String> getExtraFile() {
        if (extraFile == null) {
            extraFile = new ArrayList<String>();
        }
        return this.extraFile;
    }

    /**
     * Gets the value of the mtm property.
     * 
     * @return
     *     possible object is
     *     {@link MTMType }
     *     
     */
    public MTMType getMtm() {
        return mtm;
    }

    /**
     * Sets the value of the mtm property.
     * 
     * @param value
     *     allowed object is
     *     {@link MTMType }
     *     
     */
    public void setMtm(MTMType value) {
        this.mtm = value;
    }

    /**
     * Gets the value of the loopStructure property.
     * 
     * @return
     *     possible object is
     *     {@link LoopStructureType }
     *     
     */
    public LoopStructureType getLoopStructure() {
        return loopStructure;
    }

    /**
     * Sets the value of the loopStructure property.
     * 
     * @param value
     *     allowed object is
     *     {@link LoopStructureType }
     *     
     */
    public void setLoopStructure(LoopStructureType value) {
        this.loopStructure = value;
    }

    /**
     * Gets the value of the hardwareDependency property.
     * 
     * @return
     *     possible object is
     *     {@link HardwareDependencyType }
     *     
     */
    public HardwareDependencyType getHardwareDependency() {
        return hardwareDependency;
    }

    /**
     * Sets the value of the hardwareDependency property.
     * 
     * @param value
     *     allowed object is
     *     {@link HardwareDependencyType }
     *     
     */
    public void setHardwareDependency(HardwareDependencyType value) {
        this.hardwareDependency = value;
    }

    /**
     * Gets the value of the faultTolerance property.
     * 
     * @return
     *     possible object is
     *     {@link FaultToleranceType }
     *     
     */
    public FaultToleranceType getFaultTolerance() {
        return faultTolerance;
    }

    /**
     * Sets the value of the faultTolerance property.
     * 
     * @param value
     *     allowed object is
     *     {@link FaultToleranceType }
     *     
     */
    public void setFaultTolerance(FaultToleranceType value) {
        this.faultTolerance = value;
    }

    /**
     * Gets the value of the parentTask property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getParentTask() {
        return parentTask;
    }

    /**
     * Sets the value of the parentTask property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setParentTask(String value) {
        this.parentTask = value;
    }

    /**
     * Gets the value of the cflags property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getCflags() {
        return cflags;
    }

    /**
     * Sets the value of the cflags property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setCflags(String value) {
        this.cflags = value;
    }

    /**
     * Gets the value of the description property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the value of the description property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setDescription(String value) {
        this.description = value;
    }

    /**
     * Gets the value of the file property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getFile() {
        return file;
    }

    /**
     * Sets the value of the file property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setFile(String value) {
        this.file = value;
    }

    /**
     * Gets the value of the hasInternalStates property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getHasInternalStates() {
        return hasInternalStates;
    }

    /**
     * Sets the value of the hasInternalStates property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setHasInternalStates(YesNoType value) {
        this.hasInternalStates = value;
    }

    /**
     * Gets the value of the hasMTM property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getHasMTM() {
        return hasMTM;
    }

    /**
     * Sets the value of the hasMTM property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setHasMTM(String value) {
        this.hasMTM = value;
    }

    /**
     * Gets the value of the hasSubGraph property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getHasSubGraph() {
        return hasSubGraph;
    }

    /**
     * Sets the value of the hasSubGraph property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setHasSubGraph(String value) {
        this.hasSubGraph = value;
    }

    /**
     * Gets the value of the id property.
     * 
     * @return
     *     possible object is
     *     {@link BigInteger }
     *     
     */
    public BigInteger getId() {
        return id;
    }

    /**
     * Sets the value of the id property.
     * 
     * @param value
     *     allowed object is
     *     {@link BigInteger }
     *     
     */
    public void setId(BigInteger value) {
        this.id = value;
    }

    /**
     * Gets the value of the isHardwareDependent property.
     * 
     * @return
     *     possible object is
     *     {@link YesNoType }
     *     
     */
    public YesNoType getIsHardwareDependent() {
        return isHardwareDependent;
    }

    /**
     * Sets the value of the isHardwareDependent property.
     * 
     * @param value
     *     allowed object is
     *     {@link YesNoType }
     *     
     */
    public void setIsHardwareDependent(YesNoType value) {
        this.isHardwareDependent = value;
    }

    /**
     * Gets the value of the language property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLanguage() {
        return language;
    }

    /**
     * Sets the value of the language property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLanguage(String value) {
        this.language = value;
    }

    /**
     * Gets the value of the ldflags property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getLdflags() {
        return ldflags;
    }

    /**
     * Sets the value of the ldflags property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setLdflags(String value) {
        this.ldflags = value;
    }

    /**
     * Gets the value of the name property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the value of the name property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setName(String value) {
        this.name = value;
    }

    /**
     * Gets the value of the runCondition property.
     * 
     * @return
     *     possible object is
     *     {@link RunConditionType }
     *     
     */
    public RunConditionType getRunCondition() {
        return runCondition;
    }

    /**
     * Sets the value of the runCondition property.
     * 
     * @param value
     *     allowed object is
     *     {@link RunConditionType }
     *     
     */
    public void setRunCondition(RunConditionType value) {
        this.runCondition = value;
    }

    /**
     * Gets the value of the subGraphProperty property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getSubGraphProperty() {
        return subGraphProperty;
    }

    /**
     * Sets the value of the subGraphProperty property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setSubGraphProperty(String value) {
        this.subGraphProperty = value;
    }

    /**
     * Gets the value of the taskType property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getTaskType() {
        return taskType;
    }

    /**
     * Sets the value of the taskType property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setTaskType(String value) {
        this.taskType = value;
    }

}
