
package hopes.cic.xml;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for PortMapType complex type.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;complexType name="PortMapType"&gt;
 *   &lt;complexContent&gt;
 *     &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType"&gt;
 *       &lt;attribute name="task" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="port" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="childTask" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="childTaskPort" type="{http://peace.snu.ac.kr/CICXMLSchema}NameType" /&gt;
 *       &lt;attribute name="direction" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}PortDirectionType" /&gt;
 *       &lt;attribute name="type" use="required" type="{http://peace.snu.ac.kr/CICXMLSchema}PortMapTypeType" /&gt;
 *     &lt;/restriction&gt;
 *   &lt;/complexContent&gt;
 * &lt;/complexType&gt;
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "PortMapType")
public class PortMapType {

    @XmlAttribute(name = "task", required = true)
    protected String task;
    @XmlAttribute(name = "port", required = true)
    protected String port;
    @XmlAttribute(name = "childTask")
    protected String childTask;
    @XmlAttribute(name = "childTaskPort")
    protected String childTaskPort;
    @XmlAttribute(name = "direction", required = true)
    protected PortDirectionType direction;
    @XmlAttribute(name = "type", required = true)
    protected PortMapTypeType type;

    /**
     * Gets the value of the task property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getTask() {
        return task;
    }

    /**
     * Sets the value of the task property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setTask(String value) {
        this.task = value;
    }

    /**
     * Gets the value of the port property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getPort() {
        return port;
    }

    /**
     * Sets the value of the port property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setPort(String value) {
        this.port = value;
    }

    /**
     * Gets the value of the childTask property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getChildTask() {
        return childTask;
    }

    /**
     * Sets the value of the childTask property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setChildTask(String value) {
        this.childTask = value;
    }

    /**
     * Gets the value of the childTaskPort property.
     * 
     * @return
     *     possible object is
     *     {@link String }
     *     
     */
    public String getChildTaskPort() {
        return childTaskPort;
    }

    /**
     * Sets the value of the childTaskPort property.
     * 
     * @param value
     *     allowed object is
     *     {@link String }
     *     
     */
    public void setChildTaskPort(String value) {
        this.childTaskPort = value;
    }

    /**
     * Gets the value of the direction property.
     * 
     * @return
     *     possible object is
     *     {@link PortDirectionType }
     *     
     */
    public PortDirectionType getDirection() {
        return direction;
    }

    /**
     * Sets the value of the direction property.
     * 
     * @param value
     *     allowed object is
     *     {@link PortDirectionType }
     *     
     */
    public void setDirection(PortDirectionType value) {
        this.direction = value;
    }

    /**
     * Gets the value of the type property.
     * 
     * @return
     *     possible object is
     *     {@link PortMapTypeType }
     *     
     */
    public PortMapTypeType getType() {
        return type;
    }

    /**
     * Sets the value of the type property.
     * 
     * @param value
     *     allowed object is
     *     {@link PortMapTypeType }
     *     
     */
    public void setType(PortMapTypeType value) {
        this.type = value;
    }

}
