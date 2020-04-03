
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for CICAlgorithmType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="CICAlgorithmType">
 *   &lt;complexContent>
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *       &lt;sequence>
 *         &lt;element name="tasks" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskListType"/>
 *         &lt;element name="libraries" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryListType" minOccurs="0"/>
 *         &lt;element name="channels" type="{http://peace.snu.ac.kr/CICXMLSchema}ChannelListType" minOccurs="0"/>
 *         &lt;element name="multicastGroups" type="{http://peace.snu.ac.kr/CICXMLSchema}MulticastGroupListType" minOccurs="0"/>
 *         &lt;element name="portMaps" type="{http://peace.snu.ac.kr/CICXMLSchema}PortMapListType" minOccurs="0"/>
 *         &lt;element name="libraryConnections" type="{http://peace.snu.ac.kr/CICXMLSchema}LibraryConnectionListType" minOccurs="0"/>
 *         &lt;element name="taskGroups" type="{http://peace.snu.ac.kr/CICXMLSchema}TaskGroupListType" minOccurs="0"/>
 *         &lt;element name="modes" type="{http://peace.snu.ac.kr/CICXMLSchema}ModeListType"/>
 *         &lt;element name="headers" type="{http://peace.snu.ac.kr/CICXMLSchema}HeaderListType" minOccurs="0"/>
 *       &lt;/sequence>
 *       &lt;attribute name="property" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" />
 *     &lt;/restriction>
 *   &lt;/complexContent>
 * &lt;/complexType>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "CICAlgorithmType", propOrder = {
    "tasks",
    "libraries",
    "channels",
    "multicastGroups",
    "portMaps",
    "libraryConnections",
    "taskGroups",
    "modes",
    "headers"
})
public class CICAlgorithmType {

    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected TaskListType tasks;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected LibraryListType libraries;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected ChannelListType channels;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected MulticastGroupListType multicastGroups;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected PortMapListType portMaps;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected LibraryConnectionListType libraryConnections;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected TaskGroupListType taskGroups;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema", required = true)
    protected ModeListType modes;
    @XmlElement(namespace = "http://peace.snu.ac.kr/CICXMLSchema")
    protected HeaderListType headers;
    @XmlAttribute(required = true)
    protected String property;

    /**
     * Gets the value of the tasks property.
     * 
     * @return
     *     possible object is
     *     {@link TaskListType }
     *     
     */
    public TaskListType getTasks() {
        return tasks;
    }

    /**
     * Sets the value of the tasks property.
     * 
     * @param value
     *     allowed object is
     *     {@link TaskListType }
     *     
     */
    public void setTasks(TaskListType value) {
        this.tasks = value;
    }

    /**
     * Gets the value of the libraries property.
     * 
     * @return
     *     possible object is
     *     {@link LibraryListType }
     *     
     */
    public LibraryListType getLibraries() {
        return libraries;
    }

    /**
     * Sets the value of the libraries property.
     * 
     * @param value
     *     allowed object is
     *     {@link LibraryListType }
     *     
     */
    public void setLibraries(LibraryListType value) {
        this.libraries = value;
    }

    /**
     * Gets the value of the channels property.
     * 
     * @return
     *     possible object is
     *     {@link ChannelListType }
     *     
     */
    public ChannelListType getChannels() {
        return channels;
    }

    /**
     * Sets the value of the channels property.
     * 
     * @param value
     *     allowed object is
     *     {@link ChannelListType }
     *     
     */
    public void setChannels(ChannelListType value) {
        this.channels = value;
    }

    /**
     * Gets the value of the multicastGroups property.
     * 
     * @return
     *     possible object is
     *     {@link MulticastGroupListType }
     *     
     */
    public MulticastGroupListType getMulticastGroups() {
        return multicastGroups;
    }

    /**
     * Sets the value of the multicastGroups property.
     * 
     * @param value
     *     allowed object is
     *     {@link MulticastGroupListType }
     *     
     */
    public void setMulticastGroups(MulticastGroupListType value) {
        this.multicastGroups = value;
    }

    /**
     * Gets the value of the portMaps property.
     * 
     * @return
     *     possible object is
     *     {@link PortMapListType }
     *     
     */
    public PortMapListType getPortMaps() {
        return portMaps;
    }

    /**
     * Sets the value of the portMaps property.
     * 
     * @param value
     *     allowed object is
     *     {@link PortMapListType }
     *     
     */
    public void setPortMaps(PortMapListType value) {
        this.portMaps = value;
    }

    /**
     * Gets the value of the libraryConnections property.
     * 
     * @return
     *     possible object is
     *     {@link LibraryConnectionListType }
     *     
     */
    public LibraryConnectionListType getLibraryConnections() {
        return libraryConnections;
    }

    /**
     * Sets the value of the libraryConnections property.
     * 
     * @param value
     *     allowed object is
     *     {@link LibraryConnectionListType }
     *     
     */
    public void setLibraryConnections(LibraryConnectionListType value) {
        this.libraryConnections = value;
    }

    /**
     * Gets the value of the taskGroups property.
     * 
     * @return
     *     possible object is
     *     {@link TaskGroupListType }
     *     
     */
    public TaskGroupListType getTaskGroups() {
        return taskGroups;
    }

    /**
     * Sets the value of the taskGroups property.
     * 
     * @param value
     *     allowed object is
     *     {@link TaskGroupListType }
     *     
     */
    public void setTaskGroups(TaskGroupListType value) {
        this.taskGroups = value;
    }

    /**
     * Gets the value of the modes property.
     * 
     * @return
     *     possible object is
     *     {@link ModeListType }
     *     
     */
    public ModeListType getModes() {
        return modes;
    }

    /**
     * Sets the value of the modes property.
     * 
     * @param value
     *     allowed object is
     *     {@link ModeListType }
     *     
     */
    public void setModes(ModeListType value) {
        this.modes = value;
    }

    /**
     * Gets the value of the headers property.
     * 
     * @return
     *     possible object is
     *     {@link HeaderListType }
     *     
     */
    public HeaderListType getHeaders() {
        return headers;
    }

    /**
     * Sets the value of the headers property.
     * 
     * @param value
     *     allowed object is
     *     {@link HeaderListType }
     *     
     */
    public void setHeaders(HeaderListType value) {
        this.headers = value;
    }

    /**
     * Gets the value of the property property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getProperty() {
        return property;
    }

    /**
     * Sets the value of the property property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setProperty(String value) {
        this.property = value;
    }

}
